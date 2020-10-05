from time import time

from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
import numpy as np
from backup.dataset_backup_001 import load_dataset_old
from dataset import StreamDataset, data_select, data_select_mask, ParallelDataset
import torch


def accuracy(pred, label):
    pass

def init_weights(w, m='kaiming'):
    if m == 'kaiming':
        if type(w) == torch.nn.Linear:
            torch.nn.init.kaiming_normal_(w.weight)
    else:
        return


def correlation_plus_mse(pred, label, device):
    mse = torch.nn.MSELoss().to(device)
    loss = mse(pred, label) + label_correlation_loss(pred, label)
    return loss

def correlation_plus_MLSMLoss(pred, label, device):
    MLSM = torch.nn.MultiLabelSoftMarginLoss().to(device)
    loss = MLSM(pred, label) + label_correlation_loss(pred, label)
    return loss


class CorrelationMSELoss(torch.nn.Module):
    def __init__(self):
        super(CorrelationMSELoss, self).__init__()
        self.mse = torch.nn.MSELoss()
        self.correlation = CorrelationLoss()

    def forward(self, pred, label):
        return self.mse(pred, label) + self.correlation(pred, label)

class CorrelationMLSMLoss(torch.nn.Module):
    def __init__(self):
        super(CorrelationMLSMLoss, self).__init__()
        self.mlsm = torch.nn.MultiLabelSoftMarginLoss()
        self.correlation = CorrelationLoss()

    def forward(self, pred, label):
        return self.mlsm(pred, label) + self.correlation(pred, label)


class CorrelationLoss(torch.nn.Module):
    def __init__(self):
        super(CorrelationLoss, self).__init__()

    def forward(self, pred, label):
        if len(label.shape) == 1:
            pred = label.unsqueeze(0)
            pred = label.unsqueeze(0)

        # pred = torch.sigmoid(pred)
        loss_total = 0
        for i in range(label.shape[0]):
            loss = 0
            n_one = int(torch.sum(label[i]))
            n_zero = label.shape[1] - n_one
            zero_index = torch.nonzero(label[i] == 0, as_tuple=False).reshape(-1)
            nonzero_index = torch.nonzero(label[i] > 0, as_tuple=False).reshape(-1)

            if n_one == 0:
                for l in zero_index:
                    loss += torch.exp(pred[i][l] - 1)
                loss /= n_zero
            elif n_zero == 0:
                for l in nonzero_index:
                    loss += torch.exp(-pred[i][l])
                loss /= n_one
            else:
                for k in nonzero_index:
                    for l in zero_index:
                        loss += torch.exp(-(pred[i][k] - pred[i][l]))

                loss /= (n_one * n_zero)

            loss_total += loss

        return loss_total

def regularization_2(model):
    loss = 0
    for param in model.parameters():
        pass

def label_correlation_loss(pred, label):
    '''
    Same with class CorrelationLoss.
    :param pred:
    :param label:
    :return:
    '''
    if len(label.shape) == 1:
        pred = label.unsqueeze(0)
        pred = label.unsqueeze(0)

    # pred = torch.sigmoid(pred)
    loss_total = 0

    for i in range(label.shape[0]):
        loss = 0
        n_one = int(torch.sum(label[i]))
        n_zero = label.shape[1] - n_one
        zero_index = torch.nonzero(label[i] == 0, as_tuple=False).reshape(-1)
        nonzero_index = torch.nonzero(label[i] > 0, as_tuple=False).reshape(-1)
        if n_one == 0:
            for l in zero_index:
                loss += torch.exp(pred[i][l] - 1)

            loss /= n_zero
        elif n_zero == 0:
            for l in nonzero_index:
                loss += torch.exp(-pred[i][l])

            loss /= n_one
        else:
            for k in nonzero_index:
                for l in zero_index:
                    loss += torch.exp(-(pred[i][k] - pred[i][l]))

            loss /= (n_one * n_zero)
        loss_total += loss

    return loss_total

def test_train(old_model, new_model, assist_model, dataset, task_id, device):
    '''
    Test traing set.
    :param old_model:
    :param new_model:
    :param assist_model:
    :param dataset:
    :param task_id:
    :param device:
    :return:
    '''
    test_train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    pred_all = np.empty([0, dataset.data_y.shape[1]])

    for x, y in test_train_loader:
        x = x.to(device)
        y = y.to(device)
        pred1 = old_model(x)

        if task_id != 0:
            print(pred1.shape)
            print(old_model.end.get_out_dim())
            print(assist_model.get_io_dim())
            x2 = assist_model(pred1)
            pred2 = new_model(x, x2)
            pred = torch.cat([pred1, pred2], 1)
        else:
            pred = pred1

        # print("+++", y.cpu().detach().numpy())
        # print("---", pred.cpu().detach().numpy().round())
        # print()
        pred = pred.cpu().detach().numpy() > 0.5
        pred_all = np.concatenate([pred_all, pred], 0)

    # for i in range(pred_all.shape[0]):
    #     print("+++", dataset.data_y[i])
    #     print("---", pred_all[i])
    #     print()
    # print("+++", dataset.data_y[-1])
    # print("---", pred_all[-1])
    # print()
    # print(pred_all)
    print("Task {} Acc: {}".format(task_id, accuracy_score(dataset.data_y, pred_all)))
    # print()
    # print(pred_all.shape, dataset.data_y.shape)

def produce_pseudo_data(data, model, device, method='mask'):
    model.eval()
    dataset = None
    data_y = []

    # Get the predictions of the old model to be the psudo labels.
    for x in data.data_x:
        x = torch.Tensor(x).to(device)
        data_y.append(model(x).cpu().detach().numpy())

    data_y = np.array(data_y)
    if method == 'mask':
        mask = data_select_mask(data_y)
        dataset = ParallelDataset(data.data_x, mask, data_y.round(), data.task_id, None)

        preds = []
        reals = []
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i][j] == 1:
                    preds.append(data_y[i][j].round())
                    temp = data.all_y[i][: 7]
                    reals.append(temp[j])
        print(mask.shape[1]*mask.shape[0], np.sum(mask), accuracy_score(np.array(reals), np.array(preds)))

    else:
        selected = data_select(data.data_x, data_y, -1)  # use inter or final to find suitable samples
        mask = np.ones((data.data_x.shape[0], model.get_out_dim()))

        # Fine tune the old model by psudo labels.
        if len(selected) != 0:
            # todo how about no data.
            selected_x = []
            selected_y = []
            selected_truth = []  # test selected performance

            for t in selected:
                selected_x.append(data.data_x[t])
                selected_y.append(data_y[t].round())
                # selected_truth.append(train_set.all_y[t][: 7]) # test selected performance

            dataset = ParallelDataset(np.array(selected_x), mask, np.array(selected_y), data.task_id, None)

        # selected_y = np.array(selected_y) > 0.5 # test selected performance
        # selected_truth = np.array(selected_truth) # test selected performance
        # print(selected_y.shape, selected_truth.shape) # test selected performance
        # print("The selected accuracy is", accuracy_score(selected_truth, selected_y), accuracy_score(selected_truth.reshape(-1), selected_y.reshape(-1))) # test selected performance
    return dataset

def main():
    loss = label_correlation_loss(
        torch.Tensor([[0, 1, 0], [1, 0, 1]]),
        torch.Tensor([[0, 1, 0], [1, 0, 1]])
    )
    print(loss)
    return
    train_X, train_Y, train_Y_rest, test_X, test_Y, test_Y_rest = load_dataset_old('yeast', 103, 0.5)
    train_X = StreamDataset(train_X, train_Y, 0)
    print(train_X.data_x.shape)
    for x, y in train_X:
        print(x.shape, y.shape)


if __name__ == '__main__':
    main()
