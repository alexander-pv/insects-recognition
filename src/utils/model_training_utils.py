
"""
Набор функций и классов для работы model_training.py
"""

import os
import tqdm
import torch
import torch.nn.functional as F

import datetime
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset

from sklearn.metrics import classification_report

from utils import nn_utils, image_utils, augmentation


def get_computing_device():
    print(f'CUDA available: {torch.cuda.is_available()}')
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_pytorch_model(resnet_name, pretrained):
    return torch.hub.load('pytorch/vision:v0.6.0', resnet_name, pretrained=pretrained)


def get_optimizer(optimizer_name, model, optimizer_kwargs):
    if optimizer_name == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), **optimizer_kwargs[optimizer_name])

    elif optimizer_name == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), **optimizer_kwargs[optimizer_name])

    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs[optimizer_name])

    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs[optimizer_name])

    elif optimizer_name == 'adamax':
        optimizer = torch.optim.Adamax(model.parameters(), **optimizer_kwargs[optimizer_name])

    elif optimizer_name == 'rms_prop':
        optimizer = torch.optim.RMSprop(model.parameters(), **optimizer_kwargs[optimizer_name])

    else:
        raise NotImplementedError('Not implemented loss or text mistake.')

    return optimizer


def get_classification_report(y_true, y_pred, target_names):
    out_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    out_txt = classification_report(y_true, y_pred, target_names=target_names, output_dict=False)
    return out_dict, out_txt


class PytorchDataset(Dataset):
    """Custom dataset class for insects datasets"""

    def __init__(self, dataset_type, kwargs):
        """
        Example kwargs:
        {'class_dict': {'ant': 0, 'bee': 1},
         'dataset_rootdir': r'/home/project/dataset_rootdir',
         'seed':42,
         'val_size': 0.3,
         'test_size': 0.3
         }
        """
        super(PytorchDataset, self).__init__()
        self.kwargs = kwargs
        self.dataset_type = dataset_type
        self.transform = self.kwargs['torch_transform']

        print(f'Pytorch {self.dataset_type.upper()} dataset kwargs:'.center(50, '='))
        print(''.join(list(f'{k}: {v}\n' for k, v in self.kwargs.items())))

        def get_dataset_metadata(kwargs):
            """
            Загрузка таблицы с метаданными по датасету в директории root_datadir.
            """
            _dataset_rootdir = kwargs['dataset_rootdir']
            _class_dict = kwargs['class_dict']

            df_metadata = pd.read_csv(os.path.join(_dataset_rootdir, 'df_img_meta.csv'))
            df_metadata = df_metadata[['reg_img_path', 'class', 'sha1']]

            unique_class_labels = df_metadata['class'].unique().tolist()
            print(f'Unique classes in {_dataset_rootdir}:\n{unique_class_labels}')
            print(f'Mapping class dictionary: {_class_dict}')
            df_metadata['class'] = df_metadata['class'].map(_class_dict)
            return df_metadata

        def split_data_balanced(x, y, test_size, seed):
            """
            Сбалансированная разбивка датасета на train и test
            """
            # assert test_size < 1, 'Fraction of test_size < 1'
            np.random.seed(seed)
            unique_classes = y.unique().tolist()

            train_idx, test_idx = [], []

            # Пройдемся по каждому лейблу класса, возьмем индексы и перемешаем
            for class_label in unique_classes:
                class_data_idx = y[y == class_label].index.tolist()
                np.random.shuffle(class_data_idx)
                # Найдем необходимую линию разграничения
                threshold_idx = int(np.ceil(len(class_data_idx) * test_size))
                # Обновим train_idx и test_idx
                test_idx.extend(class_data_idx[:threshold_idx])
                train_idx.extend(class_data_idx[threshold_idx:])

            x_test = x[x.index.isin(test_idx)].copy(deep=True)
            y_test = y[y.index.isin(test_idx)].copy(deep=True)

            x_train = x[x.index.isin(train_idx)].copy(deep=True)
            y_train = y[y.index.isin(train_idx)].copy(deep=True)

            return x_train, x_test, y_train, y_test

        def split_dataset_from_metadata(df_metadata, kwargs):
            """
            Формирование train, val, test на основе таблицы с метаданными.
            Воспроизводимость поддерживается значением генератора псевдослучайных чисел.
            """

            # Splitting data to train and validation+test
            df_path_train, df_path_valtest, df_y_train, df_y_valtest = \
                split_data_balanced(df_metadata['reg_img_path'], df_metadata['class'],
                                    test_size=kwargs['val_size'], seed=kwargs['seed'])

            # Splitting validation+test data to validation and test
            df_path_val, df_path_test, df_y_val, df_y_test = \
                split_data_balanced(df_path_valtest, df_y_valtest,
                                    test_size=kwargs['test_size'], seed=kwargs['seed'])

            return df_y_train.index.tolist(), df_y_val.index.tolist(), df_y_test.index.tolist()

        # Get dataset metadata
        self.df_metadata = get_dataset_metadata(kwargs)
        # Get dataset split indicies
        traind_idx, val_idx, test_idx = split_dataset_from_metadata(self.df_metadata, self.kwargs)
        self.data_type_idx = {'train': traind_idx, 'val': val_idx, 'test': test_idx}
        # Get necessary dataset
        current_data_idx = self.data_type_idx[self.dataset_type]
        print(f'Number of observations for {self.dataset_type.upper()}: {len(current_data_idx)}\n')
        self.df_current_dataset = self.df_metadata[self.df_metadata.index.isin(current_data_idx)].copy(deep=True)
        self.df_current_dataset.reset_index(inplace=True, drop=True)

        print('TRAIN classes: ', self.df_metadata[
            self.df_metadata.index.isin(self.data_type_idx['train'])]['class'].nunique())
        print(self.df_metadata[
                  self.df_metadata.index.isin(self.data_type_idx['train'])]['class'].value_counts())

        print('VAL classes: ', self.df_metadata[
            self.df_metadata.index.isin(self.data_type_idx['val'])]['class'].nunique())
        print(self.df_metadata[
                  self.df_metadata.index.isin(self.data_type_idx['val'])]['class'].value_counts())

        print('TEST classes: ', self.df_metadata[
            self.df_metadata.index.isin(self.data_type_idx['test'])]['class'].nunique())
        print(self.df_metadata[
                  self.df_metadata.index.isin(self.data_type_idx['test'])]['class'].value_counts())

    def __len__(self):
        return self.df_current_dataset.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get registered image path and read
        img_path = self.df_current_dataset.loc[idx, 'reg_img_path']
        # Get image sha1
        img_sha1 = self.df_current_dataset.loc[idx, 'sha1']
        # Pillow opens image with RGB dimension sequence
        # image = Image.open(img_path)
        # New version for abspath data registering fix.
        image = Image.open(os.path.join(self.kwargs['dataset_rootdir'].split('dataset')[0], img_path))
        # Get class_label
        class_label = self.df_current_dataset.loc[idx, 'class']

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'class': class_label, 'img_sha1': img_sha1}
        return sample


class SaveBestModelCallback:

    def __init__(self, model_name, class_dict, save_path, info_txt):
        super(SaveBestModelCallback, self).__init__()
        self.current_best_val_loss = np.inf
        self.model_name = model_name
        self.class_dict = class_dict
        self.save_path = save_path
        self.info_txt = info_txt
        self.callback_init_time = \
            str(datetime.datetime.now()).replace(':', '_').replace('-', '_').replace(' ', '_')[:19]
        self.weights_folder = f'{self.callback_init_time}_pytorch_{self.model_name}'

        if self.weights_folder not in os.listdir(self.save_path):
            os.mkdir(os.path.join(self.save_path, self.weights_folder))

        with open(os.path.join(self.save_path, self.weights_folder,
                               f'pytorch_{self.model_name}_info.txt'), 'a') as f:
            f.write(f'Model [pytorch_{self.model_name}_epoch_...] classes dictionary: ' +
                    self.class_dict.__repr__() + '\n\n' + self.info_txt
                    )

    def check_loss_improvement(self, net, epoch, val_loss):

        if self.current_best_val_loss:
            if val_loss < self.current_best_val_loss:
                full_save_path = os.path.join(self.save_path, self.weights_folder,
                                              f'pytorch_{self.model_name}_epoch_{epoch}.pth')
                torch.save(net.state_dict(), full_save_path)
                print('[SaveBestModelCallback] ' + \
                      f'val_loss was improved: {self.current_best_val_loss} -> {val_loss}. Model was saved.')
                self.current_best_val_loss = val_loss

    def validate(self, net, criterion, epoch, val_loader, target_names, device):

        y_true, y_pred = [], []
        logits_array = None

        with torch.no_grad():
            val_loss = 0.0
            for i, val_batch in enumerate(tqdm.tqdm(val_loader)):

                # Get validation batch
                inputs, labels = val_batch['image'], val_batch['class']
                y_true.extend(labels.cpu().numpy().tolist())
                # Send data to device
                inputs, labels = inputs.to(device), labels.to(device)
                # Forward pass
                outputs = net(inputs)
                val_loss += criterion(outputs, labels)
                # Collect predictions
                _, indicies = torch.max(outputs, 1)
                y_pred.extend(indicies.cpu().numpy().tolist())

                # Collect logits array for AUC
                outputs_proba = F.softmax(outputs, dim=1)
                if isinstance(logits_array, np.ndarray):
                    logits_array = np.vstack([logits_array, outputs_proba.cpu().numpy()])
                else:
                    logits_array = outputs_proba.cpu().numpy()

            # Check val_loss and save model if necessary
            self.check_loss_improvement(net, epoch, val_loss)

        report_out_dict, report_out_txt = get_classification_report(y_true, y_pred, target_names)
        _, _, roc_auc_dict = nn_utils.compute_roc_auc(y_true, logits_array, self.class_dict)

        torch.cuda.empty_cache()
        return val_loss, report_out_dict, report_out_txt, roc_auc_dict


def model_train(net, training_kwargs, verbose=0, display_step=50, neptune_class=None):
    optimizer = training_kwargs['optimizer']
    criterion = training_kwargs['criterion']
    scheduler = training_kwargs['lr_scheduler']
    device = training_kwargs['device']

    save_best_callback = SaveBestModelCallback(model_name=training_kwargs['model_name'],
                                               class_dict=training_kwargs['class_dict'],
                                               save_path=training_kwargs['save_path'],
                                               info_txt=training_kwargs.__repr__(),
                                               )
    # Инициализация эксперимента с neptune
    if neptune_class:
        neptune_class.init_client()
        neptune_class.create_experiment()

    # Send model to device
    net.to(device)
    for epoch in range(1, training_kwargs['total_epochs'] + 1):
        torch.cuda.empty_cache()
        net.train()
        running_loss = 0.0
        t0 = datetime.datetime.now()

        for i, train_batch in enumerate(tqdm.tqdm(training_kwargs['data_loaders']['train_loader'])):
            i += 1
            # Get train batch
            inputs, labels = train_batch['image'], train_batch['class']
            # Send data to device
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()                                # Set gradients to zero
            outputs = net(inputs)                                # Forward pass
            loss = criterion(outputs, labels)                    # Calculate loss
            loss.backward()                                      # Backward pass
            optimizer.step()                                     # Make gradient step
            running_loss += loss.item()                          # Update total epoch loss

            if verbose:
                if i % display_step == 0:
                    print(f'[Training] [Epoch: {epoch}. Batch: {i}] ' +
                          f'[sum_loss: {round(running_loss, 3)} mean_loss: {round(running_loss / i, 3)}'
                          )

        # Get validation loss and other metrics
        net.eval()
        val_loss, cls_report_dict, cls_report, roc_auc_dict = \
            save_best_callback.validate(net,
                                        criterion, epoch,
                                        training_kwargs['data_loaders']['val_loader'],
                                        training_kwargs['target_names'], device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        t1 = datetime.datetime.now()

        print(f'[{(t1 - t0).total_seconds()} sec.][Epoch {epoch}] ' +
              f'train_loss: {running_loss}, val_loss: {val_loss}, learning_rate: {current_lr}.')
        print(cls_report)

        if neptune_class:
            # Логгирование данных с neptune
            # Эпоха, ошибка, шаг градиентного спуска
            neptune_class.log_metric('epoch', epoch)
            neptune_class.log_metric('train_loss', running_loss)
            neptune_class.log_metric('val_loss', val_loss.item())
            neptune_class.log_metric('learning_rate', current_lr)

            # Precision, recall, F1-score
            for weight_type in ['macro avg', 'weighted avg']:
                neptune_class.log_metric(f'{weight_type}_precision',
                                         cls_report_dict[weight_type]['precision'])
                neptune_class.log_metric(f'{weight_type}_recall',
                                         cls_report_dict[weight_type]['recall'])
                neptune_class.log_metric(f'{weight_type}_f1_score',
                                         cls_report_dict[weight_type]['f1-score'])

            # AUC ROC
            for auc_type in list(roc_auc_dict.keys()):
                neptune_class.log_metric(f'AUC_{auc_type}', roc_auc_dict[auc_type])

            neptune_class.log_text('Classification report', f'Epoch: {epoch}\n{cls_report_dict.__repr__()}')

    if neptune_class:
        neptune_class.finish_experiment()

