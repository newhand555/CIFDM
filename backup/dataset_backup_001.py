import arff
import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class StreamDataset(Dataset):
    def __init__(self, data_x, data_y, task_id, transform=None):
        self.data_x = data_x
        self.data_y = data_y
        self.task_id = task_id
        self.label_num = data_y.shape[1]
        self.transform = transform

    def get_task_id(self):
        return self.task_id

    def get_label_num(self):
        return self.label_num

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = self.data_y[idx]

        if self.transform:
            x, y = self.transform(x, y)

        return x, y


class ParallelDataset(Dataset):
    def __init__(self, data_x, data_x2, data_y, task_id, transform=None):
        self.data_x = data_x
        self.data_x2 = data_x2
        self.data_y = data_y
        self.task_id = task_id
        self.label_num = data_y.shape[1]
        self.transform = transform

    def get_task_id(self):
        return self.task_id

    def get_label_num(self):
        return self.label_num

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        x1 = self.data_x[idx]
        x2 = self.data_x2[idx]
        y = self.data_y[idx]

        if self.transform:
            x1, x2, y = self.transform(x1, x2, y)

        return x1, x2, y


def make_data_dict(data, attri_num, label_list, is_train):
    data_dict = {}
    data_dict[-1] = data[:, :attri_num]
    rest_index = attri_num

    for i in range(len(label_list)):

        if is_train:
            data_dict[i] = data[:, rest_index: rest_index + label_list[i]]
        else:
            data_dict[i] = data[:, : rest_index + label_list[i]]

        rest_index += label_list[i]

    return data_dict


def make_dataset_list(data, attri_num, label_list, instance_list, is_train):
    data_dict = make_data_dict(data, attri_num, label_list, is_train)
    data_list = []
    data_index = 0

    for i in range(len(instance_list)):
        temp_data = StreamDataset(
            data_dict[-1][data_index: data_index + instance_list[i]],
            data_dict[i][data_index: data_index + instance_list[i]],
            i,
            None
        )
        data_list.append(temp_data)
        data_index += instance_list[i]

    return data_list


def load_dataset(data_name, attri_num, label_list, train_instance_list, test_instance_list):
    if data_name == "yeast":
        train_path = 'data/yeast-train.arff'
        test_path = 'data/yeast-test.arff'
        train_data = arff.load(open(train_path, 'rt'))
        train_data = np.array(train_data['data']).astype(np.float32)
        test_data = arff.load(open(test_path, 'rt'))
        test_data = np.array(test_data['data']).astype(np.float32)
    else:
        print("The dataset {} is not supported.".format(data_name))
        return None

    total = attri_num

    for t in label_list:
        total += t

    if total > train_data.shape[1]:
        print("Error feature number.")
        return

    total = 0

    for t in train_instance_list:
        total += t

    if total > train_data.shape[0]:
        print('Error train intance number.')
        return

    total = 0

    for t in test_instance_list:
        total += t

    if total > train_data.shape[0]:
        print('Error test intance number.')
        return

    train_list = make_dataset_list(train_data, attri_num, label_list, train_instance_list, True)
    test_list = make_dataset_list(test_data, attri_num, label_list, test_instance_list, False)

    return train_list, test_list


def data_select(data_x, data_y, select_num):
    in_x = []
    in_y = []
    out_x = []
    out_y = []

    for i in range(data_y.shape[0]):
        for y in data_y[i]:
            if 0.3 < y < 0.7:
                out_x.append(data_x[i])
                out_y.append(data_y[i])
                break
        else:  # this else is for break for
            in_x.append(data_x[i])
            in_y.append(data_y[i])

    in_x = np.array(in_x)
    in_y = np.array(in_y)
    out_x = np.array(out_x)
    out_y = np.array(out_y)

    if (select_num == -1) or (in_x.shape[0] == 0):
        return in_x, in_y
    elif out_x.shape[0] == 0:
        shuffled_i = np.random.permutation(in_x.shape[1])[: select_num]
        selected_x = []
        selected_y = []

        for i in shuffled_i:
            selected_x.append(in_x[i])
            selected_y.append(in_y[i])

        return np.array(selected_x), np.array(selected_y)

    in_label_list = []
    out_label_list = []

    for j in range(data_y.shape[1]):
        temp0 = []
        temp1 = []

        for i in range(in_y.shape[0]):
            if in_y[i][j] < 0.5:
                temp0.append(in_x[i])
            else:
                temp1.append(in_x[i])

        temp0 = np.array(temp0)
        temp1 = np.array(temp1)
        in_label_list.append([temp0, temp1])
        temp0 = []
        temp1 = []

        for i in range(out_y.shape[0]):
            if out_y[i][j] < 0.5:
                temp0.append(out_x[i])
            else:
                temp1.append(out_x[i])

        temp0 = np.array(temp0)
        temp1 = np.array(temp1)
        out_label_list.append([temp0, temp1])

    print(in_label_list)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(out_label_list)

    selected = [i for i in range(len(data_x))]
    return np.array(selected)


def split_label_old(data, past_label_num_ratio):  # full-label Y data, past_label_ratio
    np.random.seed(95)
    shuffled_indices = np.random.permutation(data.shape[1])
    past_label_indices = shuffled_indices[:int(data.shape[1] * past_label_num_ratio)]
    new_label_indices = shuffled_indices[int(data.shape[1] * past_label_num_ratio) + 1:]
    print(data.shape, type(past_label_indices))
    return data[:, past_label_indices], data[:, new_label_indices]


def load_dataset_old(data_name='yeast', attri_num=103, ratio=0.5):
    if data_name == "yeast":
        train_path = 'data/yeast-train.arff'
        test_path = 'data/yeast-test.arff'
    else:
        print("The dataset {} is not supported.".format(data_name))
        return None

    train_data = arff.load(open(train_path, 'rt'))
    train_data = np.array(train_data['data']).astype(np.float32)
    train_x = train_data[:, :attri_num]
    train_y_full = train_data[:, attri_num:]
    train_y_0, train_y_1 = split_label_old(train_y_full, ratio)

    test_data = arff.load(open(test_path, 'rt'))
    test_data = np.array(test_data['data']).astype(np.float32)
    test_x = test_data[:, :attri_num]
    test_y_full = test_data[:, attri_num:]
    test_y_0, test_y_1 = split_label_old(test_y_full, ratio)
    # print(len(train_x[0]), len(train_y_0[0]), len(train_y_1[0]), len(train_y_full[0]), len(train_data[0]))

    return train_x, train_y_0, train_y_1, test_x, test_y_0, test_y_1


def main():
    data_x = np.array(
        [[1, 2, 3, 4, 5, 6],
         [1.1, 2.3, 3.5, 4.2, 5.1, 6.9],
         [1, 2, 6, 1, 4, 5],
         [8, 4, -1, 4, 5.4, 1],
         [6, 4, -1, 4, 5.4, 1],
         [7, 4, -1, 4, 5.4, 1]]
    )
    data_y = np.array(
        [[1, 0],
         [0, 1],
         [1, 1],
         [0.55, 1],
         [0.35, 1],
         [0.45, 1]]
    )
    print(data_x.shape, data_y.shape)
    data_select(data_x, data_y, 1)
    return
    train_list, test_list = load_dataset('yeast', 103, [6, 4, 4], [500, 500, 500], [300, 300, 300])
    for ds in train_list:
        print(ds.data_x.shape, ds.data_y.shape)
    for ds in test_list:
        print(ds.data_x.shape, ds.data_y.shape)


if __name__ == '__main__':
    main()
