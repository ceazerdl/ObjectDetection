import torch
import torch.nn as nn
# import get_model as models
import torchvision.models as models


def get_googlenet(path_state_dict, device, num_classes=1000, vis_model=False):
    model = models.googlenet()
    if path_state_dict:
        pretrained = torch.load(path_state_dict)
        model.load_state_dict(pretrained)

    # 更改模型
    if num_classes != 1000:
        # 主fc层
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        # 4a层后面的辅助fc层为aux1
        num_features1 = model.aux1.fc2.in_features
        model.aux1.fc2 = nn.Linear(num_features1, num_classes)
        # 4d后面的辅助fc层为aux2
        num_features2 = model.aux2.fc2.in_features
        model.aux2.fc2 = nn.Linear(num_features2, num_classes)

    if vis_model:
        from torchsummary import summary
        summary(model, input_size=(3, 224, 224), device="cpu")

    model.to(device)

    return model


