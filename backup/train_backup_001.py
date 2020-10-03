import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from dataset import data_select, StreamDataset, ParallelDataset
import numpy as np


def train_single(model, train_set, test_set, device, epoch=1):
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), 0.001)
    criterion = torch.nn.MSELoss().to(device)

    for e in range(epoch):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        # if (e+1) % 20 == 0:
        #     print(loss)
    model.eval()

    for x, y in train_loader:
        x = x.to(device)
        print(torch.round(model(x)))

    if test_set is not None:
        test_loader = DataLoader(test_set)


def train_joint(model_old, model_new, model_assist, train_set, test_set, device, epoch=1):
    # todo modify list append to find soft label
    model_old.to(device)
    model_new.to(device)
    optimizer_old = torch.optim.Adam(model_old.parameters(), 0.001)
    criterion_old = torch.nn.MSELoss().to(device)
    optimizer_assit = torch.optim.Adam(model_assist.parameters(), 0.001)
    criterion_assit = torch.nn.MSELoss().to(device)
    optimizer_new = torch.optim.Adam(model_new.parameters(), 0.001)
    criterion_new = torch.nn.MSELoss().to(device)

    for e in range(epoch):
        model_old.eval()
        data_i = []

        for x in train_set.data_x:
            x = torch.Tensor(x).to(device)
            data_i.append(model_old(x).cpu().detach().numpy())

        # TODO Test numpy with list.
        selected = data_select(train_set.data_x, train_set.data_y)  # use inter or final to find suitable samples
        selected_x = []
        selected_i = []
        selected_y = []

        for t in selected:
            selected_x.append(train_set.data_x[t])
            selected_i.append(data_i[t])
            selected_y.append(train_set.data_y[t])

        dataset_old = StreamDataset(np.array(selected_x), np.array(selected_i), train_set.task_id, None)
        dataset_assist = StreamDataset(np.array(selected_i), np.array(selected_y), train_set.task_id, None)
        model_old.train()
        train_loader = DataLoader(dataset_old, batch_size=16, shuffle=True)

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer_old.zero_grad()
            output = model_old(x)
            loss = criterion_old(output, y)
            loss.backward()
            optimizer_old.step()

        model_old.eval()
        model_assist.train()
        train_loader = DataLoader(dataset_assist, batch_size=16, shuffle=True)

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer_assit.zero_grad()
            output = model_assist(x)
            loss = criterion_assit(output, y)
            loss.backward()
            optimizer_assit.step()

        model_assist.eval()
        model_new.train()
        data_x2 = []

        for x1 in train_set.data_x:
            data_x2.append(model_assist(x1))

        dataset_new = ParallelDataset(train_set.data_x, np.array(data_x2), train_set.data_y)
        train_loader = DataLoader(dataset_new, batch_size=16, shuffle=True)
        model_new.train()

        for x1, x2, y in train_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            optimizer_new.zero_grad()
            output = model_new(x1, x2)
            loss = criterion_new(output, y)
            loss.backward()
            optimizer_new.step()

        model_new.eval()


def teacher_train_student(teacher_model, old_model, new_model, train_set, device):
    teacher_front = teacher_model.front
    teacher_end = teacher_model.end
    old_front = old_model.front
    old_end = old_model.end
    new_front = new_model.front

    temp_inter = []
    temp_label = []

    teacher_front.eval()

    for x in train_set.data_x:
        inter = teacher_front(x)
        label = teacher_end(inter)
        temp_inter.append(inter)
        temp_label.append(label)

    train_inter = StreamDataset(train_set.data_x, np.array(temp_inter), train_set.task_id, None)
    train_label = StreamDataset(np.array(temp_inter), np.array(temp_label), train_set.task_id, None)

    train_single(old_front, train_inter, None, device, 1)
    train_single(new_front, train_inter, None, device, 1)
    train_single(old_end, train_label, None, device, 1)


def train_intermedia(old_model, new_model, inter_model, train_set, device):
    pass


def student_train_teacher(teacher_model, old_model, new_model, inter_model, train_set, device):
    teacher_front = teacher_model.front
    teacher_end = teacher_model.end
    old_front = old_model.front
    old_end = old_model.end
    new_front = new_model.front
    new_end = new_model.end

    train_intermedia(old_model, new_model, inter_model, train_set, device)

    temp_input = []
    temp_inter = []
    temp_label = []
    # todo temp dataset

    for x in train_set.data_x:
        inter_old = old_front(x)
        inter_old = inter_model(inter_old)
        label_old = old_end(inter_old)

        inter_new = new_model(x)
        inter_new = inter_model(inter_new)
        label_new = new_end(inter_new)

        label = label_old.cat(label_new)

        temp_input.append(x)
        temp_inter.append(inter_old)
        temp_label.append(label)

        temp_input.append(x)
        temp_inter.append(inter_new)
        temp_label.append(label)

    train_inter = StreamDataset(np.array(temp_input), np.array(temp_inter), train_set.task_id, None)
    train_label = StreamDataset(np.array(temp_inter), np.array(temp_label), train_set.task_id, None)

    # todo try use teacher output or student output

    train_single(teacher_front, train_inter, None, device, 1)
    train_single(teacher_end, train_label, None, device, 1)
