import numpy as np
import torch

# a = [[0, 1.6, 2.4, 3, 4, 5, 6, -0.7, -0.1], [0, 1.6, 2.4, 3, 4, 5, 6, -0.7, -0.1]]
#
# a = np.array(a)
# w = np.ones(2)*2
# a = torch.Tensor(a)
# w = torch.Tensor(w)
# a = torch.transpose(a, 0, 1)
# print(a)
# print(a.shape, w.shape)
# print(torch.transpose(torch.mul(a, w), 0, 1))
# print(torch.sum(a))
#
# print(torch.div(a, w))
#
# print(a-w)

a = [[1, 2, 3], [4, 5, 6]]
b = [[1, 2, 3]]
w = torch.ones([3, 1])
a = np.array(a)
a = torch.Tensor(a)
b = np.array(b)
b = torch.Tensor(b)
a = torch.unsqueeze(a, 1)

print(a, w)
# print(torch.matmul(w, b))
print(torch.matmul(w, a))
c = 1
print(-c, -c>0)