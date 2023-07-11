import os
import torch
from torch import nn
from torchvision import datasets, models, transforms

__all__ = ['MakeModel']


def MakeModel(num_classes=10, backend='resnet50', loss_type="distance"):
    if backend not in models.list_models():
        assert False, f"Module {backend} is not available. Visit https://pytorch.org/vision/master/models.html for list of available models"

    net = getattr(models, backend)(weights="DEFAULT")
    num_features = net.fc.in_features

    if loss_type == 'distance':
        front = nn.Linear(num_features, num_classes)
        loss_fn = nn.MSELoss()
    else:
        front = nn.Sequential(nn.Linear(num_features, num_classes), nn.Softmax())
        loss_fn = nn.CrossEntropyLoss()
    for param in net.parameters():
        param.requires_grad = False

    net.fc = front
    return net, loss_fn


if __name__ == "__main__":
    model, loss_fn = MakeModel()
    x = torch.randn((255, 3, 224, 224))
    y = model(x)
    print(y.shape, x.shape)
