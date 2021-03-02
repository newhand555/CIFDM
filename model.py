import torch
import configuration
from torch import nn

class OldFrontModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(OldFrontModel, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = nn.Sequential(
            nn.Linear(in_dim, out_dim, True),
            nn.Dropout(0.1, inplace=False),
            nn.ReLU()
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
            nn.Linear(in_dim, 20, bias=True),
            nn.Dropout(0.1, inplace=False),
            nn.ReLU(),
        )
        self.out_layer = nn.Sequential(
            nn.Linear(20, out_dim, bias=True),
            # MyActivation(),
            # nn.Sigmoid(),
            nn.Sigmoid()
        )

    def modify_out_layer(self, output):
        out_dict = self.out_layer.state_dict()
        self.out_dim = output
        self.out_layer = nn.Sequential(
            nn.Linear(20, self.out_dim, bias=True),
            # MyActivation(),
            nn.Sigmoid()
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
            nn.Linear(in_dim, out_dim, True),
            nn.Dropout(0.1, inplace=False),
            nn.ReLU()
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
            nn.Linear(in_dim, 20, bias=True),
            nn.Dropout(0.1, inplace=False),
            nn.ReLU(),
        )
        self.out_layer = nn.Sequential(
            nn.Linear(20, self.out_dim, bias=True),
            # MyActivation(),
            nn.Sigmoid()
        )

    def modify_out_layer(self, output):
        self.out_dim = output
        self.out_layer = nn.Sequential(
            nn.Linear(20, output, bias=True),
            # MyActivation(),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
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

    def get_out_dim(self):
        return self.out_dim

class AssistEndModel(nn.Module):
    def __init__(self, in_dim, psudo_dim, out_dim):
        super(AssistEndModel, self).__init__()
        self.in_dim = in_dim
        self.psudo_dim = psudo_dim
        self.out_dim = out_dim
        self.layers = nn.Sequential(
            nn.Linear(self.in_dim+self.psudo_dim, self.out_dim, bias=True),
            nn.Relu(),
            nn.BatchNorm1d(self.in_dim+self.psudo_dim),
        )

    def modify_psudo_layer(self, psudo_dim):
        self.psudo_dim = psudo_dim
        self.layers = nn.Sequential(
            nn.Linear(self.in_dim + self.psudo_dim, self.out_dim, bias=True),
            nn.Relu(),
            nn.BatchNorm1d(self.in_dim + self.psudo_dim),
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], 1)
        y = self.layers(x)
        return y


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
    def __init__(self, front, inter, end):
        super(ConcatNewModel, self).__init__()
        self.front = front
        self.inter = inter
        self.end = end

    def forward(self, x1, x2):
        x1 = self.front(x1)
        x = self.inter(x1, x2)
        y = self.end(x)
        return y

    def get_out_dim(self):
        return self.end.get_out_dim()

class InterEndModel(nn.Module):
    def __init__(self, inter, end):
        super(InterEndModel, self).__init__()
        self.inter = inter
        self.end = end

    def forward(self, x1, x2):
        x = self.inter(x1, x2)
        y = self.end(x)
        return y


class ConcatTeacherModel(nn.Module):
    def __init__(self, front, end):
        super(ConcatTeacherModel, self).__init__()
        self.front = front
        self.end = end

    def forward(self, x):
        x = self.front(x)
        y = self.end(x)
        return y

    def get_out_dim(self):
        return self.end.get_out_dim()

class ConcatAssistModel(nn.Module):
    def __init__(self, front, end):
        super(ConcatAssistModel, self).__init__()
        self.front = front
        self.end = end

    def modify_io_dim(self, input, output):
        self.front.modify_io_dim(input, output)
        self.end.modify_psudo_layer(output)

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

class MySigmoid(nn.Module):
    def __init__(self, c):
        super(MySigmoid, self).__init__()
        self.params = nn.Parameter(torch.randn(c))

    def forward(self, x):
        x = x * self.params
        x = 1.0 / (1.0 + torch.exp(-x))
        return x


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