import torch.nn as nn
import numpy as np
import torch
import torchvision.models as models


def get_googlenet(path_state_dict, device, vis_model=False):
    """
    创建模型，加载参数
    :param path_state_dict:
    :param device:
    :param vis_model:
    :return:
    """
    model = models.googlenet(init_weights=False)
    if path_state_dict:
        pretrained_state_dict = torch.load(path_state_dict)
        model.load_state_dict(pretrained_state_dict)

    if vis_model:
        from torchsummary import summary
        summary(model, input_size=(3, 224, 224), device="cpu")

    model.to(device)
    return model


def generate_data(num_samples):

    # training data
    x = np.linspace(-7, 10, num_samples)[:, np.newaxis]
    print(x.shape)
    noise = np.random.normal(0, 2, x.shape)
    y = np.square(x) - 5 + noise

    # test data
    test_x = np.linspace(-7, 10, 200)[:, np.newaxis]
    noise = np.random.normal(0, 2, test_x.shape)
    test_y = np.square(test_x) - 5 + noise

    # to_tensor
    train_x, train_y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
    test_x, test_y = torch.from_numpy(test_x).float(), torch.from_numpy(test_y).float()

    return train_x, train_y, test_x, test_y



class Net(nn.Module):
    def __init__(self, n_hidden, act_func, batch_normalization=False):
        super(Net, self).__init__()
        self.do_bn = batch_normalization
        self.fcs = []    # 利用list管理网络层，forward时，方便for循环调用每个网络层
        self.bns = []
        self.bn_input = nn.BatchNorm1d(1, momentum=0.5)  # for input data
        self.act_func = act_func
        self.n_hidden = n_hidden

        # build hidden layers and BN layers
        for i in range(n_hidden):
            input_size = 1 if i == 0 else 10
            fc = nn.Linear(input_size, 10)  # 设置网络层
            setattr(self, "fc%i" % i, fc)   # set layer to the Module

            self._set_init(fc)  # params initialization1
            self.fcs.append(fc)

            if self.do_bn:
                bn = nn.BatchNorm1d(10, momentum=0.5)
                setattr(self, "bn%i" % i, bn)        # set layer to the Module
                self.bns.append(bn)

        self.predict = nn.Linear(10, 1)
        self._set_init(self.predict)

    def _set_init(self, layer):
        nn.init.normal_(layer.weight, mean=0., std=.1)
        nn.init.constant_(layer.bias, -0.2)           # B_INT = -0.2 # use a bad bias constant initializer

    def forward(self, x):

        pre_activation = [x]
        if self.do_bn:
            x = self.bn_input(x)

        layer_input = [x]

        for i in range(self.n_hidden):
            x = self.fcs[i](x)
            pre_activation.append(x)

            if self.do_bn:
                x = self.bns[i](x)

            x = self.act_func(x)

            layer_input.append(x)

        out = self.predict(x)
        return out, layer_input, pre_activation
























