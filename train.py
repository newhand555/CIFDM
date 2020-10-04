from time import time
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from dataset import data_select, StreamDataset, ParallelDataset
import numpy as np
from model import ConcatOldModel
from tool import CorrelationMLSMLoss, test_train, produce_pseudo_data


def train_single(model, train_set, test_set, device, criterion, epoch=1):
    '''
    Normal method to train a given model

    :param model: The given model.
    :param train_set: Training set.
    :param test_set: Testing set.
    :param device: CPU or GPU.
    :param epoch: Epoch number to train.
    :return: None
    '''
    train_loader = DataLoader(train_set, batch_size=1, shuffle=False)
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-08)
    # criterion = torch.nn.MSELoss().to(device)

    for e in range(epoch):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        # print("+++", y.cpu().detach().numpy())
        # print("---", output.cpu().detach().numpy().round())
        # print()
        #
        # print(e, loss, end=', ')
        # test_train(model, None, None, test_set, train_set.task_id, device)

        # if (e+1) % 20 == 0:
        #     print(loss)
    model.eval()

    for x, y in train_loader:
        x = x.to(device)

    if test_set is not None:
        test_train(model, None, None, test_set, train_set.task_id, device)


def train_joint(model_old, model_new, model_assist, train_set, test_set, device, epoch=1):
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
    criterion_new = torch.nn.MSELoss().to(device)

    # todo there are two ways: 1. train one by one. 2. train mixed.
    for e in range(epoch):
        '''
        data_y = []

        # Get the predictions of the old model to be the psudo labels.
        for x in train_set.data_x:
            x = torch.Tensor(x).to(device)
            data_y.append(model_old(x).cpu().detach().numpy())

        data_y = np.array(data_y)
        selected = data_select(train_set.data_x, data_y, -1)  # use inter or final to find suitable samples

        # Fine tune the old model by psudo labels.
        if len(selected) != 0:
            # todo how about no data.
            selected_x = []
            selected_y = []
            selected_truth = [] # test selected performance

            for t in selected:
                selected_x.append(train_set.data_x[t])
                selected_y.append(data_y[t].round())
                # selected_truth.append(train_set.all_y[t][: 7]) # test selected performance

            dataset_old = StreamDataset(np.array(selected_x), np.array(selected_y), train_set.task_id, None)

            # selected_y = np.array(selected_y) > 0.5 # test selected performance
            # selected_truth = np.array(selected_truth) # test selected performance
            # print(selected_y.shape, selected_truth.shape) # test selected performance
            # print("The selected accuracy is", accuracy_score(selected_truth, selected_y), accuracy_score(selected_truth.reshape(-1), selected_y.reshape(-1))) # test selected performance

            temp_criterion = CorrelationMLSMLoss()
            # train_single(model_old, dataset_old, None, device, temp_criterion, 1)  # epoch is 1 to train once.
        '''
        dataset_old = produce_pseudo_data(train_set, model_old, device)
        loader_old = DataLoader(dataset_old, batch_size=1, shuffle=False)
        model_old.train()
        optimizer_old = torch.optim.Adam(model_old.parameters(), weight_decay=1e-08)
        criterion = CorrelationMLSMLoss().to(device)

        for x, m, y in loader_old:
            x = x.to(device)
            m = m.to(device)
            y = torch.mul(y.to(device), m)
            optimizer_old.zero_grad()
            output = torch.mul(model_old(x), m)
            loss = criterion(output, y)
            loss.backward()
            optimizer_old.step()

        model_old.eval()
        data_x = []

        # Get outputs of old model to be the inputs of assistant model.
        for x in train_set.data_x:
            x = torch.Tensor(x).to(device)
            data_x.append(model_old(x).cpu().detach().numpy())

        data_x = np.array(data_x)
        dataset_assist = StreamDataset(data_x, train_set.data_y, train_set.task_id, None)
        temp_criterion = CorrelationMLSMLoss()
        # train assistant model once.
        train_single(model_assist, dataset_assist, None, device, temp_criterion, 1)  # must be 1

        model_assist.eval()
        data_y = []

        # Get outputs of assistant model to be the inputs.
        for x in data_x:
            x = torch.Tensor(x).to(device)
            data_y.append(model_assist(x).cpu().detach().numpy())

        data_y = np.array(data_y)
        # The dataset of new model contains two inputs, one original x, another is predictions of assistant model.
        dataset_new = ParallelDataset(train_set.data_x, data_y, train_set.data_y, train_set.task_id, None)
        loader_new = DataLoader(dataset_new, batch_size=16, shuffle=True)

        # Train new model.
        model_new.train()

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


def teacher_train_student(teacher_model, old_model, new_model, train_set, device, epoch=1):
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

    data_inter = []
    data_y = []

    teacher_front.eval()

    # Get intermedia outputs and final outputs of teacher model as targets.
    for x in train_set.data_x:
        x = torch.Tensor(x).to(device)
        inter = teacher_front(x.unsqueeze(0))
        y = teacher_end(inter)
        data_inter.append(inter.squeeze(0).cpu().detach().numpy())
        data_y.append(y.squeeze(0).cpu().detach().numpy())

    # todo use teacher front out put to be input of old end or not.
    train_inter = StreamDataset(train_set.data_x, np.array(data_inter), train_set.task_id, None)
    train_label = StreamDataset(np.array(data_inter), np.array(data_y), train_set.task_id, None)
    old_end.modify_out_layer(teacher_end.get_out_dim())

    # Use intermedia output to train front parts of both old and new models. Use final outputs to train whole old model by knowledge distillation.
    temp_criterion = torch.nn.MSELoss()
    train_single(old_front, train_inter, None, device, temp_criterion, epoch)
    train_single(new_front, train_inter, None, device, temp_criterion, epoch)
    temp_criterion = CorrelationMLSMLoss()
    train_single(old_end, train_label, None, device, temp_criterion, epoch)

    print("Teacher train student passed.")


def train_intermedia(old_model, new_model, inter_model, train_set, device):
    '''
    No use now.

    :param old_model:
    :param new_model:
    :param inter_model:
    :param train_set:
    :param device:
    :return: None
    '''
    pass


def student_train_teacher(teacher_model, train_set, device, epoch=1):
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

    for x in train_set.data_x:
        data_x.append(x)
        data_x.append(x)
        x = torch.Tensor(x).to(device)
        temp = old_front(x)
        data_inter.append(temp.cpu().detach().numpy())
        data_inter.append(new_front(x).cpu().detach().numpy())
        data_y.append(old_end(temp).cpu().detach().numpy())

    train_inter = StreamDataset(np.array(data_x), np.array(data_inter), train_set.task_id, None)
    data_y = np.concatenate((np.array(data_y), train_set.data_y), 1)
    train_all = StreamDataset(train_set.data_x, data_y, train_set.task_id, None)
    temp_criterion = torch.nn.MSELoss()
    # Train intermedia layer to compress information from both old and new front models.
    train_single(teacher_front, train_inter, None, device, temp_criterion)
    temp_criterion = CorrelationMLSMLoss()
    # Train whole model to has ability has same performance of combination task from both old and new models.
    train_single(teacher_model, train_all, None, device, temp_criterion)

    print("Student train teacher passed.")
