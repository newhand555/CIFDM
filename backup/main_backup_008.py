import argparse
import copy
from time import time
import numpy as np
import torch

from configuration import Config
from dataset import load_dataset
from model import OldFrontModel, OldEndModel, NewFrontModel, NewEndModel, IntermediaModel, AssistModel, ConcatOldModel, \
    ConcatNewModel, TeacherFrontModel, TeacherEndModel, ConcatTeacherModel, AssistEndModel, ConcatAssistModel
from other import AsymmetricLossOptimized
from tool import CorrelationMLSMLoss, init_weights, make_test, CorrelationMSELoss, TaskInfor, WeightCorrelationMSELoss, \
    CorrelationAsymmetricLoss, make_test_one
from train import train_single, train_joint, student_train_teacher, teacher_train_student, old_train_new, new_train_old


def main(opt):
    config = Config(opt)
    device = torch.device('cuda:1')
    print(config)
    train_list, test_train_list, test_data = load_dataset(True, config)

    print(test_data.data_y.shape)

    old_front_model = OldFrontModel(config.attri_num, config.embed_dim).to(device)
    old_end_model = OldEndModel(config.embed_dim, config.label_list[0]).to(device)
    new_front_model = NewFrontModel(config.attri_num, config.embed_dim).to(device)
    new_end_model = NewEndModel(config.embed_dim, config.label_list[1]).to(device)
    intermedia_model = IntermediaModel(config.embed_dim * 2, config.embed_dim).to(device)
    assist_model = AssistModel(config.label_list[0], config.label_list[1]).to(device)
    assist_end_model = AssistEndModel(config.embed_dim, config.label_list[1], config.embed_dim).to(device)
    assist_concate_model = ConcatAssistModel(assist_model, assist_end_model).to(device)
    old_concate_model = ConcatOldModel(old_front_model, old_end_model).to(device)
    new_concate_model = ConcatNewModel(new_front_model, new_end_model).to(device)

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
            for j in range(1):
                # print('The j is {}.'.format(j))
                temp_criterion = WeightCorrelationMSELoss(device, config.weight)#torch.nn.MSELoss()#
                temp_criterion = CorrelationMLSMLoss(device)
                temp_criterion = CorrelationAsymmetricLoss(device)
                train_single(old_concate_model, train_list[i], test_train_list[i], device, temp_criterion, config.first_batch, config.first_epoch, 16)
                print("The task {} result is following:".format(0))
                infor = TaskInfor([0], 'single')
                make_test(old_concate_model, new_concate_model, assist_concate_model, test_data, device, infor, config)

            if config.use_teacher == True:
                teacher_front_model = TeacherFrontModel(copy.deepcopy(old_front_model), copy.deepcopy(new_front_model),
                                                        intermedia_model)
                teacher_end_model = TeacherEndModel(copy.deepcopy(old_end_model), copy.deepcopy(new_end_model),
                                                    assist_model)
                teacher_concate_model = ConcatTeacherModel(teacher_front_model, teacher_end_model)
                teacher_train_student(teacher_concate_model, old_concate_model, new_concate_model, train_list[i],
                                      device, config)
            else:
                old_train_new(old_concate_model, new_concate_model, train_list[i], device, config)
        else:
            # Other tasks need to adjust old model and train new model.
            train_joint(old_concate_model, new_concate_model, assist_concate_model, train_list[i], None, device, config)
            task_list = []

            for j in range(i+1):
                task_list.append(j)

            infor = TaskInfor(task_list, 'single')
            make_test(old_concate_model, new_concate_model, assist_model, test_data, device, infor, config)

            if i != (config.task_num - 1):
                if config.use_teacher == True:
                    teacher_front_model = TeacherFrontModel(copy.deepcopy(old_front_model), copy.deepcopy(new_front_model),
                                                            intermedia_model)
                    teacher_end_model = TeacherEndModel(copy.deepcopy(old_end_model), copy.deepcopy(new_end_model),
                                                        assist_model)
                    teacher_concate_model = ConcatTeacherModel(teacher_front_model, teacher_end_model)
                    student_train_teacher(teacher_concate_model, train_list[i], device, config)
                    print('----------------------test teacher----------------------')
                    make_test_one(teacher_concate_model, test_data, device, infor, config)
                    teacher_train_student(teacher_concate_model, old_concate_model, new_concate_model, train_list[i], device, config)
                else:
                    new_train_old(old_concate_model, new_concate_model, train_list[i], device, config)
                    # print('----------------------test teacher----------------------')
                    # make_test_one(old_concate_model, test_data, device, infor, config)
                    old_train_new(old_concate_model, new_concate_model, train_list[i], device, config)

        print()

    print("======================== Final Result ========================")

    task_list = []
    for i in range(config.task_num):
        task_list.append(i)

    infor = TaskInfor(task_list, 'incremental')
    make_test(old_concate_model, new_concate_model, assist_model, test_data, device, infor, config)

if __name__ == '__main__':
    time_start = time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--log', type=str, default='/log/')
    opt = parser.parse_args()
    torch.set_printoptions(profile="full")

    main(opt)
    print(time() - time_start, 's.')
