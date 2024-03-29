from time import time
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from dataset import data_select, StreamDataset, ParallelDataset
import numpy as np
from model import ConcatOldModel
from other import AsymmetricLossOptimized
from tool import CorrelationMLSMLoss, produce_pseudo_data, CorrelationMSELoss, WeightCorrelationMSELoss, \
    CorrelationAsymmetricLoss, HybridLoss
from tqdm import tqdm


def train_single(model, train_set, test_set, device, criterion, batch_size=1, epoch=1, nw=4):
    '''
    Normal method to train a given model

    :param model: The given model.
    :param train_set: Training set.
    :param test_set: Testing set.
    :param device: CPU or GPU.
    :param epoch: Epoch number to train.
    :return: None
    '''
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=nw)
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-07)
    # criterion = torch.nn.MSELoss().to(device)

    for e in tqdm(range(epoch), desc="Train single mode Epoch"):

        for x, y in tqdm(train_loader, desc="Train single mode Batch", leave=False):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

    model.eval()

def train_joint(model_old, model_new, model_assist, train_set, test_set, device, config):
    '''
    Train both old model and new model at same time.

    :param model_old: Old model.
    :param model_new: New model.
    :param model_assist: Assistant model.
    :param train_set: Training set.
    :param test_set: Testing set.
    :param device: CPU or GPU.
    :param epoch: Epoch number to train.
    :return: None
    '''
    # todo modify list append to find soft label
    # Modify the input and output dims of assistant model to match requirements of new task.
    model_assist.modify_io_dim(model_old.end.get_out_dim(), train_set.data_y.shape[1])
    # Modify the output dim of new model to match number of labels of new task.
    model_new.end.modify_out_layer(train_set.data_y.shape[1])
    model_old.to(device)
    model_new.to(device)
    optimizer_new = torch.optim.Adam(model_new.parameters(), 0.001)
    criterion_new = CorrelationAsymmetricLoss(device).to(device)

    # todo there are two ways: 1. train one by one. 2. train mixed.
    # for e in range(config.pse_epoch):
    #     model_old.eval()
    #     dataset_old = produce_pseudo_data(train_set, model_old, device, 'mask')
    #
    #     if dataset_old is None:
    #         break
    #
    #     loader_old = DataLoader(dataset_old, batch_size=config.ssl_batch, shuffle=False, num_workers=config.num_workers)
    #     model_old.train()
    #     optimizer_old = torch.optim.Adam(model_old.parameters(), weight_decay=1e-08)
    #     criterion = CorrelationAsymmetricLoss(device).to(device)
    #
    #     for e2 in range(config.ssl_epoch):
    #         for x, m, y in loader_old:
    #             x = x.to(device)
    #             m = m.to(device)
    #             optimizer_old.zero_grad()
    #             output = model_old(x) # torch.mul(model_old(x), m)
    #             y = torch.mul(y.to(device), m) + torch.mul(output, (1-m))
    #             loss = criterion(output, y)
    #             loss.backward()
    #             optimizer_old.step()

    model_old.eval()
    temp_loader = DataLoader(train_set, batch_size=config.eval_batch, shuffle=True, num_workers=24)
    data_x = torch.empty([0, model_old.get_out_dim()]).to(device)

    # Get outputs of old model to be the inputs of assistant model.
    for x, _ in temp_loader:
        x = x.to(device)
        data_x = torch.cat([data_x, model_old(x)], 0)

    data_x = data_x.cpu().detach().numpy()
    dataset_assist = StreamDataset(data_x, train_set.data_y, train_set.task_id, None)
    temp_criterion = CorrelationAsymmetricLoss(device).to(device)
    # temp_criterion = WeightCorrelationMSELoss(device, config.weight)
    # train assistant model once.
    train_single(model_assist, dataset_assist, None, device, temp_criterion, config.as_batch, config.as_epoch, config.num_workers)  # must be 1
    model_assist.eval()
    data_y = torch.empty([0, model_assist.get_out_dim()]).to(device)
    temp_loader = DataLoader(dataset_assist, batch_size=256, shuffle=True, num_workers=24)

    # Get outputs of assistant model to be the inputs.
    for x, _ in temp_loader:
        x = x.to(device)
        data_y = torch.cat([data_y, model_assist(x)], 0)
        # data_y.append(model_assist(x).cpu().detach().numpy())

    data_y = data_y.cpu().detach().numpy()
    # The dataset of new model contains two inputs, one original x, another is predictions of assistant model.
    dataset_new = ParallelDataset(train_set.data_x, data_y, train_set.data_y, train_set.task_id, None)
    loader_new = DataLoader(dataset_new, batch_size=config.new_batch, shuffle=True, num_workers=config.num_workers)

    # Train new model.
    model_new.train()

    for e2 in range(config.new_epoch):
        for x, x2, y in loader_new:
            x = x.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            optimizer_new.zero_grad()
            output = model_new(x, x2)
            loss = criterion_new(output, y)
            loss.backward()
            optimizer_new.step()

    model_new.eval()
    print("Joint train passed.")


def teacher_train_student(teacher_model, old_model, new_model, train_set, device, config):
    '''
    Teacher model teaches old model and new model.

    :param teacher_model: The teacher.
    :param old_model: The old model.
    :param new_model: The new model.
    :param train_set: The training set.
    :param device: CPU or GPU.
    :param epoch: Epoch number to train.
    :return: None
    '''
    # Get each part of models.
    teacher_front = teacher_model.front
    teacher_end = teacher_model.end
    old_front = old_model.front
    old_end = old_model.end
    new_front = new_model.front

    data_inter = torch.empty([0, teacher_front.get_out_dim()]).to(device)
    data_y = torch.empty([0, teacher_end.get_out_dim()]).to(device)
    temp_loader = DataLoader(train_set, batch_size=config.eval_batch, shuffle=False, num_workers=24)

    teacher_front.eval()

    # Get intermedia outputs and final outputs of teacher model as targets.
    for x, _ in temp_loader:
        x = x.to(device)
        inter = teacher_front(x)
        y = teacher_end(inter)
        data_inter = torch.cat([data_inter, inter], 0)
        data_y = torch.cat([data_y, y], 0)

    data_inter = data_inter.cpu().detach().numpy()
    data_y = data_y.cpu().detach().numpy()

    # todo use teacher front out put to be input of old end or not.
    train_inter = StreamDataset(train_set.data_x, np.array(data_inter), train_set.task_id, None)
    train_label = StreamDataset(np.array(data_inter), np.array(data_y), train_set.task_id, None)
    old_end.modify_out_layer(teacher_end.get_out_dim())

    # Use intermedia output to train front parts of both old and new models. Use final outputs to train whole old model by knowledge distillation.
    temp_criterion = torch.nn.MSELoss()
    train_single(old_front, train_inter, None, device, temp_criterion, config.ts_batch, config.ts_epoch, config.num_workers)
    train_single(new_front, train_inter, None, device, temp_criterion, config.ts_batch, config.ts_epoch, config.num_workers)
    temp_criterion = torch.nn.MSELoss()
    train_single(old_end, train_label, None, device, temp_criterion, config.ts_batch, config.ts_epoch, config.num_workers)

    print("Teacher train student passed.")

def student_train_teacher(teacher_model, train_set, device, config):
    '''
    The teacher model is consisted by copies of old and new models. It will be adjust to be a real teacher model.

    :param teacher_model: Teacher mode that need to be adjusted.
    :param train_set: Training set.
    :param device: CPU or GPU.
    :param epoch: Epoch number to train.
    :return: None
    '''
    # todo front and end train together or seperately.
    teacher_front = teacher_model.front
    old_front = teacher_front.old.to(device).eval()
    old_end = teacher_model.end.old.to(device).eval()
    new_front = teacher_front.new.to(device).eval()

    # Get intermedia and final outputs of unchanged old and new labels to be ground truth.
    data_x = []
    data_inter = []
    data_y = []
    # data_x = torch.empty([0, train_set.data_x.shape[1]])
    temp_loader = DataLoader(train_set, batch_size=config.eval_batch, shuffle=True, num_workers=24)
    data_x = torch.empty([0, train_set.data_x.shape[1]]).to(device)
    data_inter = torch.empty([0, config.embed_dim]).to(device)
    data_y = torch.empty([0, old_end.get_out_dim()]).to(device)

    for x, _ in temp_loader:
        x = x.to(device)
        data_x = torch.cat([data_x, x], 0)
        data_x = torch.cat([data_x, x], 0)
        temp = old_front(x)
        data_inter = torch.cat([data_inter, temp], 0)
        data_inter = torch.cat([data_inter, new_front(x)], 0)
        data_y = torch.cat([data_y, old_end(temp)], 0)

    data_x = data_x.cpu().detach().numpy()
    data_inter = data_inter.cpu().detach().numpy()
    data_y = data_y.cpu().detach().numpy()
    train_inter = StreamDataset(np.array(data_x), np.array(data_inter), train_set.task_id, None)
    data_y = np.concatenate((np.array(data_y), train_set.data_y), 1)
    train_all = StreamDataset(train_set.data_x, data_y, train_set.task_id, None)
    temp_criterion = torch.nn.MSELoss()
    # Train intermedia layer to compress information from both old and new front models.
    train_single(teacher_front, train_inter, None, device, temp_criterion, config.st_batch, config.sti_epoch, config.num_workers)
    # temp_criterion = torch.nn.MSELoss()
    temp_criterion = HybridLoss(old_end.get_out_dim(), device).to(device)
    # Train whole model to has ability has same performance of combination task from both old and new models.
    train_single(teacher_model, train_all, None, device, temp_criterion, config.st_batch, config.ste_epoch, config.num_workers)

    print("Student train teacher passed.")
