import arff
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from sklearn.linear_model import LogisticRegression


class StreamDataset(Dataset):
    '''
    A standard dataset format for a given task.
    '''
    def __init__(self, data_x, data_y, task_id, transform=None, all_y = None):
        self.data_x = data_x
        self.data_y = data_y
        self.task_id = task_id
        self.label_num = data_y.shape[1]
        self.transform = transform
        self.all_y = all_y

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
    '''
    A dataset format that contains two inputs.
    '''
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


class TestDataset(Dataset):
    '''
    A dataset format for test.
    '''
    def __init__(self, data_x, data_y_dict, transform=None):
        self.data_x = data_x
        self.data_y_dict = data_y_dict
        self.task_id = -1
        self.data_y = data_y_dict[-1]
        self.label_num = self.data_y.shape[1]
        self.transform = transform

    def get_task_id(self):
        return self.task_id

    def get_label_num(self):
        return self.label_num

    def load_task(self, task_id):
        self.data_y = self.data_y_dict[task_id]
        self.label_num = self.data_y.shape[1]
        self.task_id = task_id

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = self.data_y[idx]

        if self.transform:
            x, y = self.transform(x, y)

        return x, y


def make_data_dict(data, attri_num, label_list, from_head):
    '''
    Assign labels to each task.

    :param data: Original data.
    :param attri_num: The number of columns for attributes.
    :param label_list: An array shows how many labels are assigned to each task.
    :param from_head: Whether the assign previous labels to the current task.
    :return: A dictionary that key is task id and returns specific labels for that task.
    '''
    data_dict = {}
    data_dict[-1] = data[:, :attri_num] # -1 means it contains features rather than labels.
    rest_index = attri_num

    for i in range(len(label_list)):
        if from_head:
            data_dict[i] = data[:, attri_num: rest_index + label_list[i]] # Current task labels also contain previous labels.
        else:
            data_dict[i] = data[:, rest_index: rest_index + label_list[i]] # Current task labels do not contain previous labels.

        rest_index += label_list[i]

    return data_dict


def make_train_dataset_list(data, attri_num, label_list, instance_list, from_head):
    '''
    To make a dataset to test training set.

    :param data: Original data.
    :param attri_num: The number of columns for attributes.
    :param label_list: An array shows how many labels are assigned to each task.
    :param instance_list: An array how many instances are assigned to each task.
    :param from_head: Whether the assign previous labels to the current task.
    :return: A dictionary that key is task id and returns specific instances with specific labels for that task.
    '''
    data_dict = make_data_dict(data, attri_num, label_list, from_head)
    data_list = []
    data_index = 0

    for i in range(len(label_list)):
        temp_data = StreamDataset(
            data_dict[-1][data_index: data_index + instance_list[i]],
            data_dict[i][data_index: data_index + instance_list[i]],
            i,
            None,
            data[data_index: data_index + instance_list[i], attri_num:]
        )
        data_list.append(temp_data)
        data_index += instance_list[i]

    return data_list


def make_test_dataset(data, attri_num, label_list):
    '''
    To make a testing dataset
     :param data: Original data.
    :param attri_num: The number of columns for attributes.
    :param label_list: An array shows how many labels are assigned to each task.
    :param instance_list: An array how many instances are assigned to each task.
    :return: A dictionary that key is task id and returns specific labels for that task.
    '''
    data_dict = make_data_dict(data, attri_num, label_list, False)
    data_x = data[:, : attri_num]
    data_dict[-1] = data[:, attri_num:] # -1 means it contains all labels.
    data_test = TestDataset(data_x, data_dict, None)

    return data_test


def load_dataset(data_name, attri_num, label_list, train_instance_list, test_instance_list, shuffle):
    '''
    Load dataset from file.
    :param data_name: The name of dataset.
    :param attri_num: The number of columns of attribute.
    :param label_list: An array shows how many labels are assigned to each task.
    :param train_instance_list: An array how many instances are assigned to each task.
    :param test_instance_list: An array how many instances are assigned to each task.
    :return: A tuple of lists. Each list contains a dataset for a task.
    '''
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

    if shuffle:
        temp_x = train_data[:, : attri_num]
        temp_y = train_data[:, attri_num:]
        np.random.seed(95)
        shuffled_indices = np.random.permutation(temp_y.shape[1])
        temp_y = temp_y[:, shuffled_indices]
        train_data = np.concatenate([temp_x, temp_y], 1)
        temp_x = test_data[:, : attri_num]
        temp_y = test_data[:, attri_num:]
        temp_y = temp_y[:, shuffled_indices]
        test_data = np.concatenate([temp_x, temp_y], 1)

    train_list = make_train_dataset_list(train_data, attri_num, label_list, train_instance_list, False)
    test_train_list = make_train_dataset_list(train_data, attri_num, label_list, train_instance_list, True)
    test_list = make_test_dataset(test_data, attri_num, label_list)

    return train_list, test_train_list, test_list


def calc_influence(original, train_set, test_set):
    '''
    No useful for now, ignore

    :param original:
    :param train_set:
    :param test_set:
    :return:
    '''
    if (len(train_set[0]) == 0) or (len(train_set[1]) == 0):
        return np.random.permutation(len(train_set[0]) + len(train_set[1]))

    data_x = []
    data_y = []

    for x in train_set[0]:
        data_x.append(original[x])
        data_y.append(-1)

    for x in train_set[1]:
        data_x.append(original[x])
        data_y.append(1)

    data_x = np.array(data_x)
    data_y = np.array(data_y)

    # todo logistic regression api
    model = LogisticRegression()
    model.fit(data_x, data_y)
    theta = model.coef_

    hessian = np.zeros((original.shape[1], original.shape[1]))

    for i in range(len(train_set)):
        pass


def data_select(data_x, data_y, select_num, confident=0.99):
    '''
    Select useful and trustable instances for SSL.

    :param data_x: Features.
    :param data_y: Labels.
    :param select_num: Number of selected instance.
    :param confident: The probability that the instance has can be trusted.
    :return: A list of index of selected instances.
    '''
    in_data = []
    out_data = []

    for i in range(data_y.shape[0]):
        for y in data_y[i]:
            if (1 - confident) < y < confident:
                out_data.append(i)
                break
        else:  # this else is for break for
            print(data_y[i].round(4))
            in_data.append(i)

    if (select_num == -1) or (len(in_data) == 0): # Currently it will return here, ignore the rest part.
        print("There are {} instances that can be trusted.".format(len(in_data)))
        return np.array(in_data)

    elif len(out_data) == 0:
        shuffled_i = np.random.permutation(len(in_data))[: select_num]
        return shuffled_i

    in_label_list = []
    out_label_list = []

    for j in range(data_y.shape[1]):
        temp0 = []
        temp1 = []

        for i in in_data:
            if data_y[i][j] < 0.5:
                temp0.append(i)
            else:
                temp1.append(i)

        in_label_list.append([temp0, temp1])
        temp0 = []
        temp1 = []

        for i in out_data:
            if data_y[i][j] < 0.5:
                temp0.append(i)
            else:
                temp1.append(i)

        out_label_list.append([temp0, temp1])

    print(in_label_list)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(out_label_list)

    selected = []
    for j in range(data_y.shape[1]):
        influences = calc_influence(data_x, in_label_list[j], out_label_list[j])

    selected = [i for i in range(len(data_x))]
    return np.array(selected)


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
