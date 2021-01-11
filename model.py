import torch
import configuration
from torch import nn

class OldFrontModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(OldFrontModel, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = nn.Sequential(
            nn.Linear(self.in_dim, 128, True),
            nn.Dropout(0.2, inplace=False),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256, True),
            nn.Dropout(0.2, inplace=False),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512, True),
            nn.Dropout(0.2, inplace=False),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256, True),
            nn.Dropout(0.2, inplace=False),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128, True),
            nn.Dropout(0.2, inplace=False),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, self.out_dim, True),
            nn.Dropout(0.2, inplace=False),
            nn.ReLU(),
            nn.BatchNorm1d(self.out_dim),
        )

    def forward(self, x):
        x = self.layers(x)
        return x

    def get_out_dim(self):
        return self.out_dim


class OldEndModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(OldEndModel, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = nn.Sequential(
            nn.Linear(self.in_dim, 128, True),
            nn.Dropout(0.2, inplace=False),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128, True),
            nn.Dropout(0.2, inplace=False),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )
        self.out_layer = nn.Sequential(
            nn.Linear(128, out_dim, bias=True),
            MyActivation()
        )

    def modify_out_layer(self, output):
        out_dict = self.out_layer.state_dict()
        self.out_dim = output
        self.out_layer = nn.Sequential(
            nn.Linear(128, output, bias=True),
            MyActivation()
        )

    def forward(self, x):
        x = self.layers(x)
        y = self.out_layer(x)
        return y

    def get_out_dim(self):
        return self.out_dim


class NewFrontModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NewFrontModel, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = nn.Sequential(
            nn.Linear(self.in_dim, 256, True),
            nn.Dropout(0.2, inplace=False),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128, True),
            nn.Dropout(0.2, inplace=False),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, self.out_dim, True),
            nn.Dropout(0.2, inplace=False),
            nn.ReLU(),
            nn.BatchNorm1d(self.out_dim),
        )

    def forward(self, x):
        x = self.layers(x)
        return x

    def get_out_dim(self):
        return self.out_dim


class NewEndModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NewEndModel, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = nn.Sequential(
            nn.Linear(self.in_dim, 128, bias=True),
            nn.Dropout(0.2, inplace=False),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )
        self.out_layer = nn.Sequential(
            nn.Linear(128 + self.out_dim, self.out_dim, bias=True),
            MyActivation()
        )

    def modify_out_layer(self, output):
        self.out_dim = output
        self.out_layer = nn.Sequential(
            nn.Linear(128 + output, output, bias=True),
            MyActivation()
        )

    def forward(self, x1, x2):
        x1 = self.layers(x1)
        x = torch.cat([x1, x2], 1)
        y = self.out_layer(x)
        return y

    def get_out_dim(self):
        return self.out_dim


class IntermediaModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(IntermediaModel, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(self.out_dim),
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], 1)
        y = self.layers(x)
        return y

    def get_out_dim(self):
        return self.out_dim

# todo assist input output modify
class AssistModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AssistModel, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim, bias=True),
            nn.Sigmoid()
        )

    def modify_io_dim(self, input, output):
        self.in_dim = input
        self.out_dim = output
        self.layers = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

    def get_io_dim(self):
        return self.in_dim, self.out_dim


class TeacherFrontModel(nn.Module):
    def __init__(self, old, new, inter):
        super(TeacherFrontModel, self).__init__()
        self.old = old
        self.new = new
        self.inter = inter

    def forward(self, x):
        x1 = self.old(x)
        x2 = self.new(x)
        y = self.inter(x1, x2)
        return y

    def get_out_dim(self):
        return self.inter.get_out_dim()


class TeacherEndModel(nn.Module):
    def __init__(self, old, new, assist):
        super(TeacherEndModel, self).__init__()
        self.old = old
        self.new = new
        self.assist = assist

    def forward(self, x):
        y1 = self.old(x)
        x2 = self.assist(y1)
        y2 = self.new(x, x2)
        y = torch.cat([y1, y2], 1)
        return y

    def get_out_dim(self):
        return self.old.get_out_dim() + self.new.get_out_dim()


class ConcatOldModel(nn.Module):
    def __init__(self, front, end):
        super(ConcatOldModel, self).__init__()
        self.front = front
        self.end = end

    def forward(self, x):
        x = self.front(x)
        x = self.end(x)
        return x

    def get_out_dim(self):
        return self.end.get_out_dim()

class ConcatNewModel(nn.Module):
    def __init__(self, front, end):
        super(ConcatNewModel, self).__init__()
        self.front = front
        self.end = end

    def forward(self, x1, x2):
        x1 = self.front(x1)
        y = self.end(x1, x2)
        return y

    def get_out_dim(self):
        return self.end.get_out_dim()


class ConcatTeacherModel(nn.Module):
    def __init__(self, front, end):
        super(ConcatTeacherModel, self).__init__()
        self.front = front
        self.end = end

    def forward(self, x):
        x = self.front(x)
        y = self.end(x)
        return y

class MyActivation(nn.Module):
    def __init__(self):
        super(MyActivation, self).__init__()

    def forward(self, x):
        y = torch.zeros(x.shape, device=x.device)
        mask_l = x < 0
        mask_m = torch.logical_and(0 <= x, x <= 1)
        mask_h = 1 < x
        # y[mask_l] = torch.masked_select(x - torch.square(0.5 * x), mask_l)
        y[mask_l] = torch.masked_select(x - 0.5 * torch.square(x), mask_l)
        y[mask_m] = torch.masked_select(x, mask_m)
        y[mask_h] = torch.masked_select(0.5 * torch.square(0.5 * x) + 0.5, mask_h)

        return y
        # if x < 0:
        #     return x - ((0.5 * x) ** 2) x - (0.5 * (x ** 2))
        # elif x <= 1:
        #     return x
        # else:
        #     return 0.5 * (x ** 2) + 0.5

def main():
    oldend = OldEndModel(24, 1)
    print(oldend.out_layer.parameters())
    for p in oldend.out_layer.parameters():
        print(p.shape)

    net_dict = oldend.state_dict()
    print(net_dict.keys())
    print(net_dict['out_layer.0.weight'])

if __name__ == '__main__':
    main()
