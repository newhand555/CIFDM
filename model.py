import torch
from torch import nn


class OldFrontModel(nn.Module):
    def __init__(self):
        super(OldFrontModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(103, 64, True),
            nn.Dropout(0.1, inplace=False),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class OldEndModel(nn.Module):
    def __init__(self, output):
        super(OldEndModel, self).__init__()
        self.out_dim = output
        self.layers = nn.Sequential(
            nn.Linear(64, 32, True),
            nn.Dropout(0.1, inplace=False),
            nn.ReLU(),
        )
        self.out_layer = nn.Sequential(
            nn.Linear(32, output, bias=True),
            nn.Sigmoid()

        )

    def modify_out_layer(self, output):
        self.out_dim = output
        self.out_layer = nn.Sequential(
            nn.Linear(32, output, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        y = self.out_layer(x)
        return y

    def get_out_dim(self):
        return self.out_dim


class NewFrontModel(nn.Module):
    def __init__(self):
        super(NewFrontModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(103, 64, True),
            nn.Dropout(0.1, inplace=False),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class NewEndModel(nn.Module):
    def __init__(self, output):
        super(NewEndModel, self).__init__()
        self.out_dim = output
        self.layers = nn.Sequential(
            nn.Linear(64, 32, bias=True),
            nn.Dropout(0.1, inplace=False),
            nn.ReLU(),
        )
        self.out_layer = nn.Sequential(
            nn.Linear(32 + output, output, bias=True),
            nn.Sigmoid()
        )

    def modify_out_layer(self, output):
        self.out_dim = output
        self.out_layer = nn.Sequential(
            nn.Linear(32 + output, output, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x1 = self.layers(x1)
        x = torch.cat([x1, x2], 1)
        y = self.out_layer(x)
        return y

    def get_out_dim(self):
        return self.out_dim


class IntermediaModel(nn.Module):
    def __init__(self):
        super(IntermediaModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 64, bias=True),
            nn.ReLU()
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], 1)
        y = self.layers(x)
        return y


# todo assist input output modify
class AssistModel(nn.Module):
    def __init__(self, input, output):
        super(AssistModel, self).__init__()
        self.in_dim = input
        self.out_dim = output
        self.layers = nn.Sequential(
            nn.Linear(input, output, bias=True),
            nn.Sigmoid()
        )

    def modify_io_dim(self, input, output):
        self.in_dim = input
        self.out_dim = output
        self.layers = nn.Sequential(
            nn.Linear(input, output, bias=True),
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
