import numpy as np
import torch

a = [[0, 1.6, 2.4, 3, 4, 5, 6, -0.7, -0.1], [0, 1.6, 2.4, 3, 4, 5, 6, -0.7, -0.1]]

a = np.array(a)
w = np.ones(a.shape)
a = torch.Tensor(a)
w = torch.Tensor(w)
print(torch.mul(a, w))
