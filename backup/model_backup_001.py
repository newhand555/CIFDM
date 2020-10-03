import torch
from torch import nn


class OldFrontModel(nn.Module):
    def __init__(self):
        super(OldFrontModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(103, 200, True),
            nn.Dropout(0.1, inplace=False),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class NewFrontModel(nn.Module):
    def __init__(self):
        super(NewFrontModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(103, 200, True),
            nn.Dropout(0.1, inplace=False),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class OldEndModel(nn.Module):
    def __init__(self, output):
        super(OldEndModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(200, 20, bias=True),
            nn.Dropout(0.1, inplace=False),
            nn.ReLU(),
            nn.Linear(20, output, bias=True)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class NewEndModel(nn.Module):
    def __init__(self, output):
        super(NewEndModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(200, 20, bias=True),
            nn.Dropout(0.1, inplace=False),
            nn.ReLU(),
        )
        self.out_layer = nn.Sequential(
            nn.Linear(25, output, bias=True)
        )

    def modify_out_layer(self, out_layer):
        self.out_layer = out_layer

    def forward(self, x1, x2):
        x1 = self.layers(x1)
        x = torch.cat([x1, x2], 1)
        y = self.out_layer(x)
        return y


class TeacherFrontModel(nn.Module):
    def __init__(self):
        super(TeacherFrontModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(103, 200, bias=True),
            nn.ReLU()
        )

    def forward(self, x):
        y = self.layers(x)
        return y


class TeacherEndModel(nn.Module):
    def __init__(self):
        super(TeacherEndModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(200, 11, bias=True),
            nn.ReLU()
        )

    def forward(self, x):
        y = self.layers(x)
        return y


class IntermediaModel(nn.Module):
    def __init__(self):
        super(IntermediaModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(400, 200, bias=True),
            nn.ReLU()
        )

    def foward(self, x1, x2):
        x = torch.cat([x1, x2], 1)
        y = self.layers(x)
        return y


class AssistModel(nn.Module):
    def __init__(self, input, output):
        super(AssistModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input, output, bias=True),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


class ConcatOldModel(nn.Module):
    def __init__(self, front, end):
        super(ConcatOldModel, self).__init__()
        self.front = front
        self.end = end

    def forward(self, x):
        x = self.front(x)
        x = self.end(x)
        return x


class ConcatNewModel(nn.Module):
    def __init__(self, front, end):
        super(ConcatNewModel, self).__init__()
        self.front = front
        self.end = end

    def forward(self, x1, x2):
        x1 = self.front(x1)
        y = self.end(x1, x2)
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


class IntegratedModel(nn.Module):
    def __init__(self, old_front, old_end, new_front, new_end, intermedia):
        super(IntegratedModel, self).__init__()
        self.old_front = old_front
        self.old_end = old_end
        self.new_front = new_front
        self.new_end = new_end
        self.intermedia = intermedia

    def forward(self, x):
        pass
