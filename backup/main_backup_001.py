import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

from configuration import Config
from model import *
from dataset import StreamDataset, load_dataset
from train import train_single, train_joint, student_train_teacher, teacher_train_student


def main(opt):
    config = Config(opt)
    device = torch.device('cuda')
    train_list, test_list = load_dataset('yeast', 103, [6, 5, 3], [500, 500, 500], [300, 300, 300])
    task_num = len(train_list)

    old_front_model = OldFrontModel()
    old_end_model = OldEndModel(output=6)
    new_front_model = NewFrontModel()
    new_end_model = NewEndModel(output=5)
    intermedia_model = IntermediaModel()
    assist_model = AssistModel(6, 5)
    teacher_front_model = TeacherFrontModel(old_front_model, new_front_model, intermedia_model)
    teacher_end_model = TeacherEndModel(old_end_model, new_end_model, assist_model)

    old_concate_model = ConcatOldModel(old_front_model, old_end_model)
    new_concate_model = ConcatNewModel(new_front_model, new_end_model)
    teacher_concate_model = ConcatTeacherModel(teacher_front_model, teacher_end_model)
    integrated_model = IntegratedModel(old_front_model, old_end_model, new_front_model, new_front_model, None)

    for i in range(len(train_list)):
        if i == 0:
            train_single(old_concate_model, train_list[i], device, 1)
            print("Phase 1 passed.")

    train_joint(old_concate_model, new_concate_model, assist_model, train_list[1], None, device)

    old_concate_model.to(device)
    data_y = []

    for x in train_list[1].data_x:
        x = torch.Tensor(x).to(device)
        data_y.append(old_concate_model(x).cpu().detach().numpy())

    data_y = np.array(data_y)
    data_y = np.concatenate([data_y, train_list[1].data_y], axis=1)

    st_dataset = StreamDataset(train_list[1].data_x, data_y, 1, None)
    student_train_teacher(teacher_concate_model, st_dataset, device, 1)

    teacher_train_student(teacher_concate_model, old_concate_model, new_concate_model, train_list[1], device)
    # for i in range(task_num):
    #
    #     if i == 0:
    #         train_single(old_front_model, train_list, test_list, device, 1)
    #         break

    # train_x, train_y_0, train_y_1, test_x, test_y_0, test_y_1 = load_dataset_old()
    # train_set = StreamDataset(train_x, train_y_0, 0)
    # test_set = StreamDataset(test_x, test_y_0, 0)

    # train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    # new_concate_model.to(device)
    # for x, y in train_loader:
    #     x1 = x.to(device)
    #     x2 = torch.rand(1, 4).to(device)
    #     new_concate_model(x1, x2)

    # train_single(old_concate_model, train_set, test_set, device)

    print(old_concate_model)


if __name__ == '__main__':
    # print_hi('PyCharm')
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--log', type=str, default='/log/')
    opt = parser.parse_args()

    main(opt)
