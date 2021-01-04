import arff
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from sklearn.linear_model import LogisticRegression


class StreamDataset(Dataset):
    '''
    A standard dataset format for a given task.
    '''

    def __init__(self, data_x, data_y, task_id, transform=None, all_y=None):
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

    def __init__(self, data_x, data_y, transform=None):
        self.data_x = data_x
        self.data_y = data_y
        self.label_num = self.data_y.shape[1]
        self.transform = transform

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


class TestDatasetOld(Dataset):
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
    data_dict[-1] = data[:, :attri_num]  # -1 means it contains features rather than labels.
    rest_index = attri_num

    for i in range(len(label_list)):
        if from_head:
            data_dict[i] = data[:,
                           attri_num: rest_index + label_list[i]]  # Current task labels also contain previous labels.
        else:
            data_dict[i] = data[:, rest_index: rest_index + label_list[
                i]]  # Current task labels do not contain previous labels.

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
    data_x = data[: 16, : attri_num]
    data_y = data[: 16, attri_num:]  # -1 means it contains all labels.
    data_test = TestDataset(data_x, data_y, None)

    return data_test


def load_dataset(shuffle, config):
    '''
    Load dataset from file.
    :param data_name: The name of dataset.
    :param attri_num: The number of columns of attribute.
    :param label_list: An array shows how many labels are assigned to each task.
    :param train_instance_list: An array how many instances are assigned to each task.
    :param test_instance_list: An array how many instances are assigned to each task.
    :return: A tuple of lists. Each list contains a dataset for a task.
    '''
    if config.data_name == "yeast":
        train_path = 'data/yeast/yeast-train.arff'
        test_path = 'data/yeast/yeast-test.arff'
        train_data = arff.load(open(train_path, 'rt'))
        train_data = np.array(train_data['data']).astype(np.float32)
        test_data = arff.load(open(test_path, 'rt'))
        test_data = np.array(test_data['data']).astype(np.float32)
    elif config.data_name == 'nuswide':
        train_path = 'data/nuswide-cVLADplus/nus-wide-full-cVLADplus-train.arff'
        test_path = 'data/nuswide-cVLADplus/nus-wide-full-cVLADplus-test.arff'
        train_data = arff.load(open(train_path, 'rt'))
        train_data = np.array(train_data['data'])
        train_data = train_data[:, 1:].astype(np.float32)
        test_data = arff.load(open(test_path, 'rt'))
        test_data = np.array(test_data['data'])
        test_data = test_data[:, 1:].astype(np.float32)
    else:
        print("The dataset {} is not supported.".format(config.data_name))
        return None

    total = config.attri_num

    for t in config.label_list:
        total += t

    if total > train_data.shape[1]:
        print(total)
        print("Error feature number.")
        return

    total = 0

    for t in config.train_instance_list:
        total += t

    if total > train_data.shape[0]:
        print('Error train intance number.')
        return

    total = 0

    # for t in config.test_instance_list:
    #     total += t

    if total > train_data.shape[0]:
        print('Error test intance number.')
        return

    if shuffle:
        temp_x = train_data[:, : config.attri_num]
        temp_y = train_data[:, config.attri_num:]
        np.random.seed(95)
        shuffled_indices = np.random.permutation(temp_y.shape[1])
        temp_y = temp_y[:, shuffled_indices]
        train_data = np.concatenate([temp_x, temp_y], 1)
        temp_x = test_data[:, : config.attri_num]
        temp_y = test_data[:, config.attri_num:]
        temp_y = temp_y[:, shuffled_indices]
        test_data = np.concatenate([temp_x, temp_y], 1)

    train_list = make_train_dataset_list(train_data, config.attri_num, config.label_list, config.train_instance_list,
                                         False)
    test_train_list = make_train_dataset_list(train_data, config.attri_num, config.label_list,
                                              config.train_instance_list, True)
    test_list = make_test_dataset(test_data, config.attri_num, config.label_list)

    return train_list, test_train_list, test_list


def data_select_mask(data_y, ci0=0.1, ci1=0.3):
    round0 = np.logical_and(-ci0 < data_y, data_y < ci0)
    round1 = np.logical_and((1 - ci1) < data_y, data_y < (1 + ci1))
    psedu = np.logical_or(round0, round1).astype(int)
    return psedu


def data_select(data_x, data_y, select_num, ci0=0.1, ci1=0.3):
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
            if (ci0 < y < (1 - ci1)) or (y < (-ci0)) or (y > (1 + ci1)):
                out_data.append(i)
                break
        else:  # this else is for break for
            # print(data_y[i].round(4))
            in_data.append(i)

    if (select_num == -1) or (len(in_data) == 0):  # Currently it will return here, ignore the rest part.
        print("There are {} instances that can be trusted.".format(len(in_data)))
        return np.array(in_data)

    elif len(out_data) == 0:
        shuffled_i = np.random.permutation(len(in_data))[: select_num]
        return shuffled_i


def main():
    pass


if __name__ == '__main__':
    main()
