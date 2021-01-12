from time import time

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from torch.utils.data import DataLoader
import numpy as np
from dataset import StreamDataset, data_select, data_select_mask, ParallelDataset
import torch

def init_weights(w, m='kaiming'):
    if m == 'kaiming':
        if type(w) == torch.nn.Linear:
            torch.nn.init.kaiming_normal_(w.weight)
    else:
        return

class TaskInfor:
    def __init__(self, task_list, method):
        self.task_list = task_list
        self.method = method

class IntervalLoss(torch.nn.Module):
    def __init__(self, loss_function):
        super(IntervalLoss, self).__init__()
        self.loss_function = loss_function()

    def forward(self, pred, label):
        loss = torch.zeros(pred.shape, device=pred.device)

        mask_round_zero = torch.logical_and(-0.2 < pred, pred < 0)
        mask_one = torch.logical_and(label == 1, mask_round_zero)
        loss[mask_one] = torch.masked_select(0.2 - (1 * pred), mask_one)

        mask_round_zero = torch.logical_and(0 <= pred, pred < 0.2)
        mask_one = torch.logical_and(label == 1, mask_round_zero)
        loss[mask_one] = torch.masked_select(0.2 - (1 * pred), mask_one)

        mask_round_one = torch.logical_and(0.8 < pred, pred < 1)
        mask_zero = torch.logical_and(label == 0, mask_round_one)
        loss[mask_zero] = torch.masked_select(0.2 * (pred - 1) + 1, mask_zero)

        mask_round_one = torch.logical_and(1 <= pred, pred < 1.2)
        mask_zero = torch.logical_and(label == 0, mask_round_one)
        loss[mask_zero] = torch.masked_select(0.2 - (1 * (pred - 1)), mask_zero)

        loss = loss.sum()
        loss += self.loss_function(pred, label)
        return loss

class CorrelationMSELoss(torch.nn.Module):
    def __init__(self, device):
        super(CorrelationMSELoss, self).__init__()
        self.mse = torch.nn.MSELoss()
        self.correlation = CorrelationLoss(device)

    def forward(self, pred, label):
        return self.mse(pred, label) + self.correlation(pred, label)

class WeightCorrelationMSELoss(torch.nn.Module):
    def __init__(self, device, weight=1):
        super(WeightCorrelationMSELoss, self).__init__()

        if weight == 1:
            self.mse = torch.nn.MSELoss()
        else:
            self.mse = WeightMSELoss(device, weight)

        self.correlation = CorrelationLoss(device, weight)

    def forward(self, pred, label):
        return self.mse(pred, label) + self.correlation(pred, label)

class WeightMSELoss(torch.nn.Module):
    def __init__(self, device, weight=1):
        super(WeightMSELoss, self).__init__()
        self.device = device
        self.weight = weight

    def forward(self, pred, label):
        loss_matrix = (pred - label) ** 2
        loss_matrix *= ((self.weight - 1) * label) + 1
        return torch.sum(loss_matrix)

class CorrelationMLSMLoss(torch.nn.Module):
    def __init__(self, device):
        super(CorrelationMLSMLoss, self).__init__()
        self.mlsm = torch.nn.MultiLabelSoftMarginLoss()
        self.correlation = CorrelationLoss(device)

    def forward(self, pred, label):
        return self.mlsm(pred, label) + self.correlation(pred, label)


class CorrelationLoss(torch.nn.Module):
    def __init__(self, device, weight=1):
        super(CorrelationLoss, self).__init__()
        self.device = device
        self.weight = weight

    def forward(self, pred, label):
        if len(label.shape) == 1:
            pred = label.unsqueeze(0)

        n_one = torch.sum(label, 1)
        n_zero = torch.ones(label.shape[0]).to(self.device) * label.shape[1]
        n_zero -= n_one

        result_matrix = torch.zeros(pred.shape).to(self.device)

        temp_result = torch.exp(pred - 1)
        temp_result = torch.transpose(temp_result, 1, 0)
        temp_n = n_zero + (n_zero == 0).float()
        temp_result = torch.div(temp_result, temp_n)
        temp_mask = (n_one == 0).float()
        temp_result = torch.mul(temp_result, temp_mask)
        temp_result = torch.transpose(temp_result, 1, 0)
        temp_result = torch.mul(temp_result, (1-label))
        result_matrix += temp_result

        temp_result = torch.exp(-pred)
        temp_result = torch.transpose(temp_result, 1, 0)
        temp_n = n_one + (n_one == 0).float()
        temp_result = torch.div(temp_result, temp_n)
        temp_mask = (n_zero == 0).float()
        temp_result = torch.mul(temp_result, temp_mask)
        temp_result = torch.transpose(temp_result, 1, 0)
        temp_result = torch.mul(temp_result, label)
        result_matrix += temp_result

        temp_result = torch.transpose(torch.matmul(torch.ones([label.shape[1], 1]).to(self.device), torch.unsqueeze(pred, 1)), 1, 2)
        temp_minus = torch.matmul(torch.ones([label.shape[1], 1]).to(self.device), torch.unsqueeze(pred, 1))
        temp_result = torch.exp(temp_minus - temp_result) * torch.unsqueeze(1-label, 1)
        temp_result = temp_result * torch.transpose(torch.unsqueeze(label, 1), 1, 2)
        temp_result = torch.sum(temp_result, 2)
        temp_result = torch.transpose(temp_result, 1, 0)
        n_else = n_one * n_zero
        temp_n = n_else + (n_else == 0).float()
        temp_result = torch.div(temp_result, temp_n)
        temp_mask = (n_else != 0).float()
        temp_result = torch.mul(temp_result, temp_mask)
        temp_result = torch.transpose(temp_result, 1, 0)
        temp_result = torch.mul(temp_result, label)
        result_matrix += temp_result

        result_matrix *= ((self.weight - 1) * label) + 1

        return torch.sum(result_matrix)

def produce_pseudo_data(data, model, device, method='mask'):
    model.eval()
    dataset = None
    data_y = torch.empty([0, model.get_out_dim()]).to(device)
    temp_loader = DataLoader(data, batch_size=256, shuffle=False, num_workers=24)

    # Get the predictions of the old model to be the psudo labels.
    for x, _ in temp_loader:
        x = x.to(device)
        data_y = torch.cat([data_y, model(x)], 0)

    data_y = data_y.cpu().detach().numpy()

    if method == 'mask':
        mask = data_select_mask(data_y)
        dataset = ParallelDataset(data.data_x, mask, data_y.round(), data.task_id, None) if np.sum(mask) != 0 else None

        preds = []
        reals = []
        counter = 0
        # for i in range(mask.shape[0]):
        #     for j in range(mask.shape[1]):
        #         if mask[i][j] == 1:
        #             preds.append(data_y[i][j].round())
        #             temp = data.all_y[i]
        #             reals.append(temp[j])
        #             if temp[j] == 1:
        #                 counter+=1
        # print('mask', mask.shape, mask.shape[1]*mask.shape[0], np.sum(mask), counter/np.sum(mask), '%', "Acc", accuracy_score(np.array(reals), np.array(preds)), "Prec", precision_score(np.array(reals), np.array(preds)), "Recall", recall_score(np.array(reals), np.array(preds)))

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
                selected_truth.append(data.all_y[t][: model.get_out_dim()]) # test selected performance

            dataset = ParallelDataset(np.array(selected_x), mask, np.array(selected_y), data.task_id, None)

            selected_y = np.array(selected_y) > 0.5 # test selected performance
            selected_truth = np.array(selected_truth) # test selected performance
            print(selected_y.shape, selected_truth.shape) # test selected performance
            print('None', data.data_x.shape[0], selected_y.shape[0], accuracy_score(selected_truth, selected_y), accuracy_score(selected_truth.reshape(-1), selected_y.reshape(-1)))
            # print("The selected accuracy is", accuracy_score(selected_truth, selected_y), accuracy_score(selected_truth.reshape(-1), selected_y.reshape(-1))) # test selected performance
    return dataset

def make_test(old_concate_model, new_concate_model, assist_model, test_data, device, infor, config):
    # todo check details and modify usage
    label_index = [0]
    for l in config.label_list:
        label_index.append(l+label_index[-1])

    test_loader = DataLoader(test_data, batch_size=config.eval_batch, shuffle=False, num_workers=4)
    old_concate_model.to(device).eval()
    new_concate_model.to(device).eval()
    assist_model.to(device).eval()

    outputs = np.empty((0, old_concate_model.get_out_dim()+new_concate_model.get_out_dim()))
    groud_truth = np.empty((0, test_data.data_y.shape[1]))

    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        pred1 = old_concate_model(x)
        x2 = assist_model(pred1)
        pred2 = new_concate_model(x, x2)
        pred = torch.cat([pred1, pred2], 1)
        outputs = np.concatenate([outputs, pred.cpu().detach().numpy()], 0)
        groud_truth = np.concatenate([groud_truth, y.cpu().detach().numpy()], 0)

    for idx in infor.task_list:
        if infor.method == 'single':
            s_idx = label_index[idx]
            e_idx = label_index[idx + 1]
        elif infor.method == 'incremental':
            s_idx = label_index[0]
            e_idx = label_index[idx + 1]
        else:
            print("Error in the function make_test.")
            exit(1)

        if infor.method == 'single':
            print("The task {} result is following:".format(idx))
        elif infor.method == 'incremental':
            print("Until the task {} result is following:".format(idx))

        real_label = groud_truth[:, s_idx: e_idx]
        pred_label = outputs[:, s_idx: e_idx]

        print("The test shape is {}.".format(real_label.shape))
        # print("Test AUC: {}".format(roc_auc_score(real_label, pred_label, average='micro')))

        real_label = np.array(real_label) > 0.5
        pred_label = np.array(pred_label) > 0.5
        print("Test Accuracy: {}, {}".format(accuracy_score(real_label, pred_label),
                                        accuracy_score(real_label.reshape(-1), pred_label.reshape(-1))))
        # print("Test AUC: {}".format(roc_auc_score(real_label.reshape(-1), pred_label.reshape(-1))))
        print("Test Precision: {}".format(precision_score(real_label.reshape(-1), pred_label.reshape(-1))))
        print("Test Recall: {}".format(recall_score(real_label.reshape(-1), pred_label.reshape(-1))))
        print()

def main():
    lossfunc = CorrelationLoss(torch.device('cpu'))
    loss = lossfunc(
        torch.Tensor([[0.1, 0.9, 0.3], [0.13, 0.87, 0.31]]),
        torch.Tensor([[0, 0, 0], [1, 0, 1]])
    )
    lossfunc = WeightMSELoss(torch.device('cpu'))
    loss = lossfunc(
        torch.Tensor([[0.1, 0.9, 0.3], [0.13, 0.87, 0.31]]),
        torch.Tensor([[0, 0, 0], [1, 0, 1]])
    )
    return


if __name__ == '__main__':
    main()
