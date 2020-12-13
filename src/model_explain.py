
import os
import tqdm
import torch
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import config

from torchvision import transforms

from lime import lime_image
from skimage.segmentation import mark_boundaries
from skimage.transform import resize

from utils import gradcam, augmentation
from utils.misc_functions import save_class_activation_images

from PIL import Image


class LimeExplainContainer:

    def __init__(self, model, resize, meta_dataset, class_dict, weights_path, device):
        super(LimeExplainContainer, self).__init__()
        self.model = model
        self.resize = resize
        self.device = device
        self.invert_class = dict([(v, k) for k, v in class_dict.items()])
        self.meta_dataset = meta_dataset
        self.lime_explainer = lime_image.LimeImageExplainer()

        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()
        self.model.to(self.device)

        self.pil_transform = transforms.Compose([augmentation.ImgAugTransform(with_aug=False),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 ]
                                                )
        self.preprocess_transform = transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize(mean=config.IMG_NORMALIZE_MEAN,
                                                                             std=config.IMG_NORMALIZE_STD)
                                                        ]
                                                       )
        self.meta_dataset.reset_index(inplace=True, drop=True)
        self.cwd = os.getcwd().replace('src', '')
        print('\nLimeExplainContainer init\n')
        if 'lime_results' not in os.listdir('../'):
            os.mkdir('../lime_results')

    def batch_predict(self, images):

        self.model.eval()
        batch = torch.stack(tuple(self.preprocess_transform(i) for i in images), dim=0)
        batch = batch.to(self.device)

        logits = self.model(batch)
        probs = torch.nn.functional.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    def generate_test_explanations(self, top_labels, num_features, hide_color=0, num_samples=2000,
                                   positive_only=False, hide_rest=False,
                                   figsize=(10, 10), save_img=False):

        for i in tqdm.tqdm(range(self.meta_dataset.shape[0])):
            row = self.meta_dataset.iloc[i]
            #  Поверка для нового старого варианта формирования отноистельных путей
            if row['reg_img_path'].split('/')[0] == 'datasets':
                img = Image.open(os.path.join(self.cwd, row['reg_img_path']))
            else:
                img = Image.open(row['reg_img_path'])

            class_label = row['class']
            print('Image: ', row['reg_img_path'])
            print(f'Class: {self.invert_class[class_label]}')

            explanation = self.lime_explainer.explain_instance(np.array(self.pil_transform(img)),
                                                               self.batch_predict,
                                                               top_labels=top_labels,
                                                               hide_color=hide_color,
                                                               num_samples=num_samples)

            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
                                                        positive_only=positive_only,
                                                        num_features=num_features,
                                                        hide_rest=hide_rest)

            img_boundry = mark_boundaries(temp / 255.0, mask)

            fig = plt.figure(figsize=figsize)
            plt.axis('off')
            plt.title(self.invert_class[class_label], fontsize=14)
            plt.imshow(img_boundry)
            if save_img:
                sha1_hash = row['reg_img_path'].split('.')[0].split(os.sep)[-1]
                fig.savefig(f'../lime_results/lime_local_explanation_{self.invert_class[class_label]}_{sha1_hash}.png')


class GradCamContainer:

    def __init__(self, model, model_name, resize, meta_dataset, class_dict, weights_path, device):
        super(GradCamContainer, self).__init__()
        self.model = model
        self.model_name = model_name
        self.resize = resize
        self.device = device
        self.invert_class = dict([(v, k) for k, v in class_dict.items()])
        self.meta_dataset = meta_dataset

        self.transform_img = transforms.Compose([augmentation.ImgAugTransform(with_aug=False),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=config.IMG_NORMALIZE_MEAN,
                                                                      std=config.IMG_NORMALIZE_STD)
                                                 ]
                                                )

        self.model.load_state_dict(torch.load(weights_path))
        self.model.to(self.device)
        self.meta_dataset.reset_index(inplace=True, drop=True)
        self.cwd = os.getcwd().replace('src', '')
        print('\nGradCamContainer init\n')

    def visualize_layers(self, img, img_sha1, target_class, layer_min, layer_max):

        img_original = img.resize((self.resize, self.resize))
        img_tensor = torch.cuda.FloatTensor(np.expand_dims(self.transform_img(img), axis=0))

        if isinstance(layer_min, str):
            assert layer_min == layer_max
            layer = layer_min
            gcam = gradcam.GradCam(self.model, self.model_name, layer)
            cam = gcam.generate_cam(img_tensor, target_class=target_class)
            save_class_activation_images(
                img_original,
                cam,
                f'gcam_{self.invert_class[target_class]}_{self.model_name}_{img_sha1}_layer_{layer}'
            )
        else:
            for layer in range(layer_min, layer_max + 1):
                gcam = gradcam.GradCam(self.model, self.model_name, layer)
                cam = gcam.generate_cam(img_tensor, target_class=target_class)
                save_class_activation_images(
                    img_original,
                    cam,
                    f'gcam_{self.invert_class[target_class]}_{self.model_name}_{img_sha1}_layer_{layer}'
                )

    def generate_heatmaps(self, layer_min=0, layer_max=18, batch_size=4):

        for i in tqdm.tqdm(range(self.meta_dataset.shape[0])):
            row = self.meta_dataset.iloc[i]
            img_sha1 = row['reg_img_path'].split('.')[0].split(os.sep)[-1]
            img_path = row['reg_img_path']
            class_label = row['class']

            if row['reg_img_path'].split('/')[0] == 'datasets':
                img = Image.open(os.path.join(self.cwd, img_path))
            else:
                img = Image.open(row['reg_img_path'])

            print('Image: ', img_path)
            print(f'Class: {self.invert_class[class_label]}')

            self.visualize_layers(img, img_sha1, class_label, layer_min, layer_max)


class RiseContainer(torch.nn.Module):
    def __init__(self, model, class_dict, meta_dataset, input_size, weights_path, device, gpu_batch):
        super(RiseContainer, self).__init__()
        self.model = model
        self.device = device
        self.meta_dataset = meta_dataset
        self.input_size = input_size
        self.weights_path = weights_path
        self.gpu_batch = gpu_batch
        self.transform_img = transforms.Compose([augmentation.ImgAugTransform(with_aug=False),
                                                 transforms.Resize(self.input_size),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=config.IMG_NORMALIZE_MEAN,
                                                                      std=config.IMG_NORMALIZE_STD)
                                                 ]
                                                )
        self.model.load_state_dict(torch.load(self.weights_path))
        self.model.eval()
        self.model.to(self.device)
        self.invert_class = dict([(v, k) for k, v in class_dict.items()])
        self.cwd = os.getcwd().replace('src', '')

        self.meta_dataset.reset_index(inplace=True, drop=True)
        if 'rise_results' not in os.listdir('../'):
            os.mkdir('../rise_results')
        print('\nRiseContainer init\n')

    def generate_masks(self, N, s, p1):
        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')

        self.masks = np.empty((N, *self.input_size))

        for i in range(N):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            self.masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                         anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        self.masks = torch.from_numpy(self.masks).float()
        self.masks = self.masks.cuda()
        self.N = N
        self.p1 = p1

    def forward(self, x):
        N = self.N
        _, _, H, W = x.size()
        # Apply array of filters to the image
        stack = torch.mul(self.masks, x.data)

        # p = nn.Softmax(dim=1)(model(stack)) processed in batches
        p = []
        for i in range(0, N, self.gpu_batch):
            with torch.no_grad():
                p.append(self.model(stack[i:min(i + self.gpu_batch, N)]))
        p = torch.cat(p)
        # Number of classes
        CL = p.size(1)
        sal = torch.matmul(p.data.transpose(0, 1), self.masks.view(N, H * W))
        sal = sal.view((CL, H, W))
        sal = sal / N / self.p1
        return sal

    def plot_rise_mask(self, img_original, class_id, rise_mask, sha1_hash, save_fig,
                       figsize=(10, 10), fontsize=14, alpha=0.7):

        fig = plt.figure(figsize=figsize)
        plt.axis('off')
        plt.title(self.invert_class[class_id], fontsize=fontsize)
        plt.imshow(img_original.resize(self.input_size))
        plt.imshow(rise_mask.cpu().numpy()[class_id], cmap='jet', alpha=alpha)

        if save_fig:
            fig.savefig(f'../rise_results/rise_local_explanation_{self.invert_class[class_id]}_{sha1_hash}.png')

    def get_rise_heatmap(self, img, class_id, sha1_hash, save_fig, n, s, p1):
        self.generate_masks(N=n, s=s, p1=p1)
        img_transformed = self.transform_img(img)
        img_tensor = torch.cuda.FloatTensor(np.expand_dims(img_transformed, axis=0))
        rise_mask = self.forward(img_tensor)

        del self.masks
        torch.cuda.empty_cache()

        self.plot_rise_mask(img, class_id, rise_mask, sha1_hash, save_fig)

    def generate_rise_heatmaps(self, save_fig=True, n=1000, s=8, p1=0.2):

        for i in tqdm.tqdm(range(self.meta_dataset.shape[0])):
            row = self.meta_dataset.iloc[i]

            if row['reg_img_path'].split('/')[0] == 'datasets':
                img = Image.open(os.path.join(self.cwd, row['reg_img_path']))
            else:
                img = Image.open(row['reg_img_path'])

            class_id = row['class']
            sha1_hash = row['reg_img_path'].split('.')[0].split(os.sep)[-1]
            print('Image: ', row['reg_img_path'])
            print(f'Class: {self.invert_class[class_id]}')
            self.get_rise_heatmap(img, class_id, sha1_hash, save_fig, n, s, p1)


