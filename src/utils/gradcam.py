"""
Created on Thu Oct 26 11:06:51 2017
@author: Utku Ozbulak - github.com/utkuozbulak
"""
from PIL import Image
import numpy as np
import torch


class CamExtractor:
    """
        Extracts cam features from the model
    """
    def __init__(self, model, model_name, target_layer):
        self.model = model
        self.model_name = model_name
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None

        if self.model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                                'wide_resnet50_2', 'wide_resnet101_2', 'resnext50_32x4d', 'resnext101_32x8d']:
            items = self.model._modules.items()
        else:
            items = self.model.features._modules.items()
        for module_pos, module in items:
            # print('module_pos, module', module_pos, module)
            if module_pos in ['avgpool', 'fc']:
                pass
            else:
                x = module(x)  # Forward
            # Target layer может быть либо число, либо строка
            try:
                module_pos = int(module_pos)
            except:
                pass
            if module_pos == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)

        # Здесь необходимо учитывать, как преобразуется выход сверток в конкретной модели
        assert self.model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                                   'wide_resnet50_2', 'wide_resnet101_2', 'resnext50_32x4d',
                                   'resnext101_32x8d', 'mobilenet_v2', 'mobilenet_v3_large']

        if self.model_name == 'mobilenet_v2':
            x = torch.nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        elif self.model_name == 'mobilenet_v3_large':
            x = self.model.conv(x)
            x = self.model.avgpool(x).reshape(x.shape[0], -1)
        elif ('resnet' in self.model_name) | ('resnext' in self.model_name):
            x = self.model.avgpool(x).reshape(x.shape[0], -1)
        else:
            x = x.view(x.size(0), -1)  # Flatten

        # Forward pass on the classifier
        if 'mobilenet' in self.model_name:
            x = self.model.classifier(x)
        else:
            x = self.model.fc(x)
        return conv_output, x


class GradCam:
    """
        Produces class activation map
    """
    def __init__(self, model, model_name, target_layer):
        self.model = model
        self.model_name = model_name
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, model_name, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.cpu().numpy())
        # Target for backprop
        one_hot_output = torch.cuda.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads

        if 'mobilenet' in self.model_name:
            self.model.features.zero_grad()
            self.model.classifier.zero_grad()
        else:
            for module_pos, module in list(self.model._modules.items()):
                module.zero_grad()

        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.cpu().numpy()[0]
        # Get convolution outputs
        target = conv_output.data.cpu().numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices with antialiasing, however,
        # when I moved the repository to PIL, this option was out of the window.
        # So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, do not hesitate to send a PR.

        # You can also use the code below instead of the code line above, suggested by @ ptschandl
        # from scipy.ndimage.interpolation import zoom
        # cam = zoom(cam, np.array(input_image[0].shape[1:])/np.array(cam.shape))
        return cam