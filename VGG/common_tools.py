import torch
import torchvision.models as models
import torch.nn as nn


def get_vgg16(path_state_dict, device, vis_model=False):
    """
    创建模型，加载参数
    :param path_state_dict:
    :param device:
    :param vis_model:
    :return:
    """
    model = models.vgg16()
    pretrained_state_dict = torch.load(path_state_dict)
    model.load_state_dict(pretrained_state_dict)

    # 本数据集为2分类，所以更改vgg模型的输出为2
    num_features = model.classifier[6].in_features
    new_linear = nn.Linear(num_features, 2)
    model.classifier._modules["6"] = new_linear

    if vis_model:
        from torchsummary import summary
        summary(model, input_size=(3, 224, 224), device="cpu")

    model.to(device)
    return model