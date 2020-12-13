
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import pandas as pd

from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Случайное семплирование из списка заданных индексов для данных с несбалансированными классами
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, class_dict, random_state, indices=None, num_samples=None,
                 verbose=True):

        np.random.seed(random_state)
        # If indices is not provided,
        # all elements in the dataset will be considered
        self.verbose = verbose
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # Define invert class dict: indices -> class names
        self.invert_class_dict = {j: i for i, j in class_dict.items()}
        # If num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
        # Distribution of classes in the dataset
        self.label_to_count = dataset.df_metadata['class'].value_counts().to_dict()
        # Invert class numeric labels to string labels
        self.label_to_count_names = {self.invert_class_dict[k]: v for k, v in self.label_to_count.items()}

        # Weight for each sample
        weights = [1.0 / self.label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

        if self.verbose:
            self.info_weights = {k: round(1/v, 3) for k, v in self.label_to_count_names.items()}
            _print_weights = ''.join([f'\n{k}: {v}' for k, v in self.info_weights.items()])
            print(f'[ImbalancedDatasetSampler] Sampler weights:\n{_print_weights}')

    def _get_label(self, dataset, idx):
        return dataset.df_current_dataset.loc[idx, 'class']

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


def xavier_init(m):
    """Xavier weight initialization
    """
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)


def he_init(m):
    """He weight initialization
    """
    if type(m) == nn.Conv2d:
        torch.nn.init.kaiming_uniform_(m.weight)
    elif type(m) == nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight)


def one_hot_encode(y, n_classes):
    y_encoded = np.eye(n_classes)
    return y_encoded[y]


def plot_multiclass_roc_curves(fpr, tpr, roc_auc, class_dict, save_roc_plot=False, data_hash=None,
                               figsize=(12, 10), lw=2, title_fontsize=14):
    """
    Plot all ROC curves
    :param fpr:
    :param tpr:
    :param roc_auc:
    :param class_dict:
    :param neptune_logger:
    :param data_hash:
    :param figsize:
    :param lw:
    :param title_fontsize:
    :return:
    """

    sns.set()
    plt.figure(figsize=figsize)
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle='--', linewidth=2)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle='-.', linewidth=2)

    colors = cycle(['green', 'darkorange', 'cornflowerblue'])
    class_names = list(class_dict.keys())
    for i, color in zip(class_names, colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic', fontsize=title_fontsize)
    plt.legend(loc="lower right")

    if save_roc_plot:
        # Check that data_hash exists
        assert data_hash is not None
        # Save plot
        plot_name = f'roc_curve_{data_hash}.png'
        plt.savefig(os.path.join('..', 'log_artifacts', 'images',  plot_name))
    else:
        plt.show()


def compute_roc_auc(y_test, y_pred, class_dict):
    """
    y_test - list of labels
    y_pred - NxC matrix of logits from pytroch model,
             N - number of observations
             C - number of classes
    """

    # Check that number of class names equal to model classes number
    assert len(list(class_dict.keys())) == y_pred.shape[1]

    n_classes = y_pred.shape[1]
    class_names = list(class_dict.keys())
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    y_hot_encoded = one_hot_encode(y_test, n_classes)

    # Compute ROC curve and ROC area for each class
    for i, class_name in enumerate(class_names):
        fpr[class_name], tpr[class_name], _ = roc_curve(y_hot_encoded[:, i], y_pred[:, i])
        roc_auc[class_name] = auc(fpr[class_name], tpr[class_name])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_hot_encoded.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[class_name] for class_name in class_names]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i, class_name in enumerate(class_names):
        mean_tpr += interp(all_fpr, fpr[class_name], tpr[class_name])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return fpr, tpr, roc_auc

