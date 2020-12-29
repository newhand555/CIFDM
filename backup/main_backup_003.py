import argparse
import copy
from time import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from configuration import Config
from dataset import load_dataset
from model import OldFrontModel, OldEndModel, NewFrontModel, NewEndModel, IntermediaModel, AssistModel, ConcatOldModel, \
    ConcatNewModel, TeacherFrontModel, TeacherEndModel, ConcatTeacherModel
from tool import CorrelationMLSMLoss, test_train, init_weights, make_test, CorrelationMSELoss, TaskInfor
from train import train_single, train_joint, student_train_teacher, teacher_train_student
from torch.utils.data import DataLoader


def main(opt):
    config = Config(opt)
    device = torch.device('cuda:0')
    train_list, test_train_list, test_data = load_dataset(True, config)
    test_train_flag = False

    old_front_model = OldFrontModel()
    old_end_model = OldEndModel(output=config.label_list[0])
    new_front_model = NewFrontModel()
    new_end_model = NewEndModel(output=config.label_list[1])
    intermedia_model = IntermediaModel()
    assist_model = AssistModel(config.label_list[0], config.label_list[1])
    old_concate_model = ConcatOldModel(old_front_model, old_end_model)
    new_concate_model = ConcatNewModel(new_front_model, new_end_model)

    # old_front_model.apply(init_weights)
    # old_end_model.apply(init_weights)
    # new_front_model.apply(init_weights)
    # new_end_model.apply(init_weights)
    # intermedia_model.apply(init_weights)
    # assist_model.apply(init_weights)

    for i in range(config.task_num):
        print("======================== Task {} ========================".format(i))

        if i == 0:
            # Task 0, only train old model as base.
            temp_criterion = CorrelationMSELoss()
            train_single(old_concate_model, train_list[i], test_train_list[i], device, temp_criterion, config.first_batch, config.first_epoch, 16)
            if test_train_flag: test_train(old_concate_model, None, None, test_train_list[i], i, device)
            print("The task {} result is following:".format(0))
            infor = TaskInfor([0], 'single')
            make_test(old_concate_model, new_concate_model, assist_model, test_data, device, infor, config)
        else:
            # Other tasks need to adjust old model and train new model.
            train_joint(old_concate_model, new_concate_model, assist_model, train_list[i], None, device, config)
            if test_train_flag: test_train(old_concate_model, new_concate_model, assist_model, test_train_list[i], i, device)
            task_list = []

            for j in range(i+1):
                task_list.append(j)

            infor = TaskInfor(task_list, 'single')
            make_test(old_concate_model, new_concate_model, assist_model, test_data, device, infor, config)

            if i != (config.task_num - 1):
                teacher_front_model = TeacherFrontModel(copy.deepcopy(old_front_model), copy.deepcopy(new_front_model),
                                                        intermedia_model)
                teacher_end_model = TeacherEndModel(copy.deepcopy(old_end_model), copy.deepcopy(new_end_model),
                                                    assist_model)
                teacher_concate_model = ConcatTeacherModel(teacher_front_model, teacher_end_model)
                student_train_teacher(teacher_concate_model, train_list[i], device, config)
                teacher_train_student(teacher_concate_model, old_concate_model, new_concate_model, train_list[i], device, config)

        print()

    print("======================== Final Result ========================")

    task_list = []
    for i in range(config.task_num):
        task_list.append(i)

    infor = TaskInfor(task_list, 'incremental')
    make_test(old_concate_model, new_concate_model, assist_model, test_data, device, infor, config)

    # print("The overall result is following:")
    # make_test(old_concate_model, new_concate_model, assist_model, test_data, device, -1, config)



if __name__ == '__main__':
    time_start = time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--log', type=str, default='/log/')
    opt = parser.parse_args()

    main(opt)
    print(time() - time_start, 's.')
