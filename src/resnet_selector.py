
import torch
import torch.nn as nn


def load_pytorch_model(resnet_name, pretrained):

    return torch.hub.load('pytorch/vision:v0.6.0', resnet_name, pretrained=pretrained)


def freeze_layers(resnet_model):

    for param in resnet_model.parameters():
        param.requires_grad = False
    return resnet_model


def get_resnet(version, class_number, pretrained, freeze_conv=False):
    """
    Выбрать версию сети ResNet
    :param version:      'resnet18','resnet34', 'resnet50', 'resnet101', 'resnet152',
                         'wide_resnet50_2', 'wide_resnet101_2',
                         'resnext50_32x4d', 'resnext101_32x8d',
    :param class_number: python int
    :param pretrained:   bool
    :param freeze_conv:  bool
    :return: torch model
    """
    model = load_pytorch_model(version, pretrained)
    print(f'Loaded: {version}')
    if freeze_conv:
        model = freeze_layers(model)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, class_number)
    return model
