import torch
import numpy as np
import torch.nn as nn

torch.manual_seed(1)
np.random.seed(1)


class MLP(nn.Module):

    def __init__(self, neural_num, layers=100, do_bn=False):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(neural_num, neural_num, bias=False) for i in range(layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(neural_num) for i in range(layers)])
        self.neural_num = neural_num
        self.do_bn = do_bn

    def forward(self, x):

        for (i, linear), bn in zip(enumerate(self.linears), self.bns):
            x = linear(x)
            if self.do_bn:
                x = bn(x)
            x = torch.relu(x)

            if torch.isnan(x.std()):
                print("output is nan in {} layers".format(i))
                break
            print("layers:{}, std:{}".format(i, x.std().item()))
        return x

    def initialize(self, mode, std_init=1):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if mode == "normal":
                    nn.init.normal_(m.weight.data, std=std_init)
                elif mode == "kaiming":
                    nn.init.kaiming_normal_(m.weight.data)
                else:
                    print("不支持{}输入".format(mode))


if __name__ == "__main__":

    # BN的作用：标准化，把数据固定在均值为0，方差为1的分布
    neural_nums = 256
    layer_nums = 100
    batch_size = 16

    # net = MLP(neural_nums, layer_nums, do_bn=False)      # 1. 无初始化； # 2. normal_初始化； # 3。 kaiming初始化
    net = MLP(neural_nums, layer_nums, do_bn=True)        # 4. BN+无初始化； 5. BN + normal; 6. BN + kaiming, 7. BN+1000
    # net.initialize("normal", std_init=1)
    # net.initialize("normal", std_init=1000)
    net.initialize("kaiming")
    inputs = torch.randn((batch_size, neural_nums))

    output = net(inputs)
    print(output)
