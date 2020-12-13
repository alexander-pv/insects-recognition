
import torch
import torch.nn as nn
import torchvision.models as models

from add import mobilenetv3


def get_mobilenet(version, class_number, pretrained, freeze_conv=False):
    """
    Выбрать версию сети MobileNet
    https://github.com/mortezamg63/Accessing-and-modifying-different-layers-of-a-pretrained-model-in-pytorch
    :param version:       mobilenet_v2, mobilenet_v3_large (5.483M parameters), mobilenet_v3_small(2.543M parameters)
    :param class_number:  python int
    :param pretrained:    bool
    :param freeze_conv:   bool
    :return: torch model
    """

    if version == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=pretrained)

        if freeze_conv:
            for param in model.parameters():
                param.requires_grad = False
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, class_number)

    elif version == 'mobilenet_v3_large':
        model = mobilenetv3.mobilenetv3_large()
        if pretrained:
            model.load_state_dict(torch.load('weights/pytorch_mobilenetv3_pretrained/mobilenetv3-large-1cd25616.pth'))

        if freeze_conv:
            for param in model.parameters():
                param.requires_grad = False
        num_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_features, class_number)

    elif version == 'mobilenet_v3_small':
        model = mobilenetv3.mobilenetv3_small()
        if pretrained:
            model.load_state_dict(torch.load('weights/pytorch_mobilenetv3_pretrained/mobilenetv3-small-55df8e1f.pth'))

        if freeze_conv:
            for param in model.parameters():
                param.requires_grad = False
        num_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_features, class_number)

    else:
        raise NotImplementedError('Only mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small are available.')
    print(f'Loaded: {version}')
    return model
