import argparse
import copy
from time import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score

from configuration import Config
from dataset import load_dataset
from model import OldFrontModel, OldEndModel, NewFrontModel, NewEndModel, IntermediaModel, AssistModel, ConcatOldModel, \
    ConcatNewModel, TeacherFrontModel, TeacherEndModel, ConcatTeacherModel
from tool import CorrelationMLSMLoss, test_train
from train import train_single, train_joint, student_train_teacher, teacher_train_student
from torch.utils.data import DataLoader


def main(opt):
    config = Config(opt)
    device = torch.device('cuda')
    # train_list, test_train_list, test_data = load_dataset('yeast', 103, [14], [1500], [900], True)
    # train_list, test_train_list, test_data = load_dataset('yeast', 103, [6, 5, 3], [500, 500, 500], [300, 300, 300], True)
    train_list, test_train_list, test_data = load_dataset('yeast', 103, [7, 6], [900, 500], [500, 500], True)
    task_num = len(train_list)
    test_train_flag = False

    old_front_model = OldFrontModel()
    old_end_model = OldEndModel(output=train_list[0].get_label_num())
    new_front_model = NewFrontModel()
    new_end_model = NewEndModel(output=train_list[1].get_label_num())
    intermedia_model = IntermediaModel()
    assist_model = AssistModel(train_list[0].get_label_num(), train_list[1].get_label_num())
    old_concate_model = ConcatOldModel(old_front_model, old_end_model)
    new_concate_model = ConcatNewModel(new_front_model, new_end_model)

    print(len(train_list))
    for data in train_list:
        print(data.data_x.shape, data.data_y.shape)
    for data in test_train_list:
        print(data.data_x.shape, data.data_y.shape)

    for i in range(task_num):
        if i == 0:
            # Task 0, only train old model as base.
            temp_criterion = CorrelationMLSMLoss()
            train_single(old_concate_model, train_list[i], test_train_list[i], device, temp_criterion, 5)
            if test_train_flag: test_train(old_concate_model, None, None, test_train_list[i], i, device)
        else:
            # Other tasks need to adjust old model and train new model.
            train_joint(old_concate_model, new_concate_model, assist_model, train_list[i], None, device, 12)
            if test_train_flag: test_train(old_concate_model, new_concate_model, assist_model, test_train_list[i], i, device)
            print("Phase {} passed.".format(i))

            if i == (task_num - 1):
                break

            teacher_front_model = TeacherFrontModel(copy.deepcopy(old_front_model), copy.deepcopy(new_front_model),
                                                    intermedia_model)
            teacher_end_model = TeacherEndModel(copy.deepcopy(old_end_model), copy.deepcopy(new_end_model),
                                                assist_model)
            teacher_concate_model = ConcatTeacherModel(teacher_front_model, teacher_end_model)
            student_train_teacher(teacher_concate_model, train_list[i], device, 12)
            teacher_train_student(teacher_concate_model, old_concate_model, new_concate_model, train_list[i], device, 12)

    onlynew = True
    test_data.load_task(task_num-1)

    if task_num == 1:
        test_data.load_task(-1)

    print(test_data.get_label_num())
    print(test_data.data_y.shape)

    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    old_concate_model.to(device).eval()
    new_concate_model.to(device).eval()
    assist_model.to(device).eval()

    outputs = np.empty((0, test_data.get_label_num()))
    real_labels = np.empty((0, test_data.get_label_num()))

    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        pred1 = old_concate_model(x)
        x2 = assist_model(pred1)
        pred2 = new_concate_model(x, x2)
        if task_num == 1:
            pred = pred1
        elif onlynew:
            pred = pred2
        else:
            pred = torch.cat([pred1, pred2], 1)

        print("+++", y.cpu().detach().numpy())
        print("---", pred.cpu().detach().numpy().round())
        print()
        outputs = np.concatenate([outputs, pred.cpu().detach().numpy().round()], 0)
        real_labels = np.concatenate([real_labels, y.cpu().detach().numpy().round()], 0)
    outputs = np.array(outputs)
    real_labels = np.array(real_labels)
    print(outputs.shape, real_labels.shape)
    print("Test Acc: {}".format(accuracy_score(real_labels, outputs)))


if __name__ == '__main__':
    time_start = time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--log', type=str, default='/log/')
    opt = parser.parse_args()

    main(opt)
    print(time() - time_start, 's.')
