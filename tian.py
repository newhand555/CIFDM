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

# a = [[1, 2, 3], [4, 5, 6]]
# b = [[1, 2, 3]]
# w = torch.ones([3, 1])
# a = np.array(a)
# a = torch.Tensor(a)
# b = np.array(b)
# b = torch.Tensor(b)
# a = torch.unsqueeze(a, 1)
#
# print(a, w)
# # print(torch.matmul(w, b))
# print(torch.matmul(w, a))
# c = 1
# print(-c, -c>0)
#
# e = 3
# d = torch.empty([0, e])
# print(a.shape)
# print(a.squeeze(0).shape)
# print(d)
# d = torch.cat([d, a.squeeze(1)], 0)
# print(d.shape)

a = [[1, 2, 3], [4, 5, 6]]
b = [1, 2, 3, 4, 5, 6]
c = [2, 2, 2, 2, 2 ,2]
w = torch.randn(3)
a = np.array(a)
a = torch.Tensor(a)
b = torch.Tensor(b)
c = torch.Tensor(c)
print(w, b)
print(w*a)
print(torch.pow(b, 2))

print(b[:3], b[3:])