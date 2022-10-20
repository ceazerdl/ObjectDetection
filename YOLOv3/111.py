import torch
import torch.nn as nn
from collections import OrderedDict


class models(nn.Module):

    def __init__(self):
        super(models, self).__init__()
        self.conv1 = nn.Sequential()
        self.conv1.add_module("conv1", nn.Conv2d(3, 64, (3, 3), (1, 1)))
        self.conv1.add_module("conv2", nn.Conv2d(64, 128, (5, 5), (1, 1)))
        self.conv2 = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 64, (3, 3), (1, 1))),
            ("conv2", nn.Conv2d(64, 128, (5, 5), (1, 1)))
        ]))
    def forward(self):
        pass


# model = models()
# print(model.conv1)
# print(model.conv1._modules["conv1"])
# print(model.conv1[0])

class A(object):

    def __init__(self):
        self.a = 1
        self.b = 2

    def need(self):
        self.c = 3
        self.d = 4

    def reneed(self):
        print(self.c + self.a)


aa = A()
# print(aa.a)
# print(aa.c)
# aa.reneed()

a = torch.tensor([[1, 2],
                  [2, 3]])
b = torch.tensor([[3, 4],
                  [4, 4]])
c = torch.cat((a, b), 0)
# print(c.shape)

a = torch.tensor([1, 2, 3])
print(a[[1, 2, 0, 2, 0]])

























