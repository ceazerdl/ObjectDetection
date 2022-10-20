import tensorflow as tf
import torch


# b = [0, 1, 1, 1, 2, 0]
# c = [2, 3, 0, 4, 6, 2]
# d = [0, 1, 0, 2, 1, 2]
# e = [0, 1, 2, 5, 4, 1]
# a = torch.ByteTensor(3, 7, 4, 6).fill_(0)
# print(a)
# print(a[0][0])
# print(a[:, 0])
# a[b, c, d, e] = 1
# print(a)

# print(tf.__version__)


# with open("1.txt", "rt", encoding="utf-8") as f:
#     t = f.readlines()

a = torch.FloatTensor([2, 4, 6, 4])
b = torch.FloatTensor([[3, 4, 5, 7],
                       [3, 4, 5, 6]])
c = 3
b[1, a > c] = 1
print(b)
# a = {'b':1, 'c':2, 'd':3}
# if 'x' not in a:
#     print("cuole")