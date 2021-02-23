
"""
Процесс проведения тестирования обученных моделей
"""
import os
import cv2
import torch
import logging
import torchvision

import hashlib
import numpy as np
import pandas as pd
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import DataLoader

from utils import nn_utils, image_utils, augmentation
from utils.model_test_utils import model_evaluate
from utils.model_training_utils import get_computing_device, load_pytorch_model, get_optimizer, \
                                       PytorchDataset, SaveBestModelCallback, model_train

import model_explain
import resnet_selector
import mobilenet_selector
import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )


def test_model(model_name, dataset_name, folder_to_evaluate, model_type):
    """
    Функция проведения оценки моделей по тестовому множеству данных
    :param model_name:      название модели
    :param dataset_name: наименование датасета
    :param folder_to_evaluate: папка с весами модели
    :param model_type: тип модели
    :return: None
    """

    assert dataset_name in config.DATASETS_LIST
    assert model_name in config.MODELS_LIST

    CONFIG = {'model': {'model_name': model_name,
                        'optimizer': config.MODEL_OPTIMIZER,
                        'pretrained': config.PRETRAINED,
                        'freeze_conv': config.FREEZE_CONV,
                        'epochs': config.EPOCHS,
                        'batch_size': config.BATCH_SIZE,
                        },

              'data': {'root_dir': os.getcwd().replace('src', 'datasets'),
                       'dataset_name': dataset_name,
                       'resize_img': config.RESIZE_IMG,
                       'num_workers': config.NUM_WORKERS,
                       'pytorch_aug': config.PYTORCH_AUG,
                       'save_aspect_ratio': config.SAVE_ASPECT_RATIO,
                       'imgaug_aug': config.IMGAUG_AUG,
                       'img_normalize': config.IMG_NORMALIZE,
                       },
              }

    # Determine random seed
    DEVICE = get_computing_device()
    CONFIG.update({'seed': config.SEED})
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    FULL_DATASET_DIR = os.path.join(CONFIG['data']['root_dir'],
                                    CONFIG['data']['dataset_name'])
    CLASS_DICT = image_utils.make_class_dict(os.path.join(FULL_DATASET_DIR, 'df_img_meta.csv'))

    if CONFIG['data']['num_workers'] > 0:
        cv2.setNumThreads(0)

    # Setting image transformations
    _pytorch_aug = CONFIG['data']['pytorch_aug']
    _img_normalize = CONFIG['data']['img_normalize']
    _save_aspect_ratio = CONFIG['data']['save_aspect_ratio']

    default_transforms_list = [augmentation.ImgAugTransform(with_aug=False),
                               image_utils.CustomResize((CONFIG['data']['resize_img'],
                                                         CONFIG['data']['resize_img']),
                                                        _save_aspect_ratio
                                                        ),
                               torchvision.transforms.ToTensor()
                               ]

    if _pytorch_aug:
        # Pytorch default & augmentation transform
        transform_list = [augmentation.ImgAugTransform(with_aug=CONFIG['data']['imgaug_aug']),
                          image_utils.CustomResize((CONFIG['data']['resize_img'],
                                                    CONFIG['data']['resize_img']),
                                                   _save_aspect_ratio
                                                   ),
                          torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
                          torchvision.transforms.RandomHorizontalFlip(),
                          torchvision.transforms.RandomRotation(20, resample=Image.BILINEAR),
                          torchvision.transforms.ToTensor()
                          ]
    else:
        transform_list = [augmentation.ImgAugTransform(with_aug=CONFIG['data']['imgaug_aug']),
                          image_utils.CustomResize((CONFIG['data']['resize_img'],
                                                    CONFIG['data']['resize_img']),
                                                   _save_aspect_ratio
                                                   ),
                          torchvision.transforms.ToTensor()
                          ]

    if _img_normalize:
        # Parameters were taken from Pytorch example for Imagenet
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L197-L198
        default_transforms_list.append(torchvision.transforms.Normalize(mean=config.IMG_NORMALIZE_MEAN,
                                                                        std=config.IMG_NORMALIZE_STD)
                                       )
    train_transforms = torchvision.transforms.Compose(transform_list)
    val_trainsforms = torchvision.transforms.Compose(default_transforms_list)

    # Set dictionaries for training, validation and testing
    TRAIN_DATASET_KWARGS = {'class_dict': CLASS_DICT,
                            'dataset_rootdir': FULL_DATASET_DIR,
                            'val_size': config.TESTVAL_SIZE,
                            'test_size': config.TEST_SIZE_FROM_TESTVAL,  # Доля от val_size
                            'seed': config.SEED,
                            'torch_transform': train_transforms,
                            }
    TESTVAL_DATASET_KWARGS = {'class_dict': CLASS_DICT,
                              'dataset_rootdir': FULL_DATASET_DIR,
                              'val_size': config.TESTVAL_SIZE,
                              'test_size': config.TEST_SIZE_FROM_TESTVAL,  # Доля от val_size
                              'seed': config.SEED,
                              'torch_transform': val_trainsforms,
                              }

    train_dataset = PytorchDataset('train', TRAIN_DATASET_KWARGS)
    test_dataset = PytorchDataset('test', TESTVAL_DATASET_KWARGS)
    test_data_loader = DataLoader(test_dataset, batch_size=CONFIG['model']['batch_size'],
                                  shuffle=True, num_workers=CONFIG['data']['num_workers'])

    n_classes = len(TESTVAL_DATASET_KWARGS['class_dict'].keys())

    if 'mobilenet' in CONFIG['model']['model_name']:
        nnet = mobilenet_selector.get_mobilenet(version=CONFIG['model']['model_name'],
                                                class_number=n_classes,
                                                pretrained=CONFIG['model']['pretrained'],
                                                freeze_conv=CONFIG['model']['freeze_conv'])
    else:
        nnet = resnet_selector.get_resnet(version=CONFIG['model']['model_name'],
                                          class_number=n_classes,
                                          pretrained=CONFIG['model']['pretrained'],
                                          freeze_conv=CONFIG['model']['freeze_conv'])

    # Train dataset img hashes to dataset hash. Universal for every machine.
    train_data_hash = \
        hashlib.sha1('_'.join(sorted(train_dataset.df_metadata['sha1'].values.tolist())).encode()).hexdigest()

    weights_folder = os.path.join(os.getcwd(), 'weights',
                                  f"""pytorch_{CONFIG['model']['model_name']}_{train_data_hash}""")
    weights_folder = os.path.join('/', *(weights_folder.split('/')[:-1] + [folder_to_evaluate]))

    EVAL_DICT = {'device': DEVICE,
                 'weights_folder': weights_folder,
                 'test_loader': test_data_loader,
                 'model': nnet,
                 'criterion': torch.nn.CrossEntropyLoss(),
                 'target_names': list(TRAIN_DATASET_KWARGS['class_dict'].keys()),
                 'class_dict': CLASS_DICT,
                 'data_hash': train_data_hash,
                 'external_data_class_dict': None,
                 }
    reports_list = model_evaluate(EVAL_DICT)

    # Формирование табличного отчета по качеству классфикации
    # Для объективной оценки испольузем только лучшую эпоху по валидации.
    assert len(reports_list) == 1
    report_dict = reports_list[0]
    df_detailed_report = pd.DataFrame([])
    report_keys = list(report_dict.keys())
    for m in ['top1_error', 'accuracy', 'macro avg', 'weighted avg', 'epoch']:
        report_keys.remove(m)

    for class_name in report_keys:
        df_detailed_report = df_detailed_report.append([{'dataset_id': train_data_hash,
                                                         'task_name': dataset_name,
                                                         'class_imbalanced_tool': model_type,
                                                         'model_name': model_name,
                                                         'class': class_name,
                                                         'precision': report_dict[class_name]['precision'],
                                                         'recall': report_dict[class_name]['recall'],
                                                         'f1': report_dict[class_name]['f1-score'],
                                                         'support': report_dict[class_name]['support'],
                                                         'best_epoch': report_dict['epoch']
                                                         }
                                                        ]
                                                       )

    if 'df_detailed_report.csv' in os.listdir(os.getcwd()):
        df_detailed_report.to_csv('df_detailed_report.csv', index=False, header=False, mode='a')
    else:
        df_detailed_report.to_csv('df_detailed_report.csv', index=False, mode='a')

    df_general_report = pd.DataFrame([{'dataset_id': train_data_hash,
                                       'task_name': dataset_name,
                                       'class_imbalanced_tool': model_type,
                                       'model_name': model_name,

                                       'weighted_avg_precision': report_dict['weighted avg']['precision'],
                                       'weighted_avg_recall': report_dict['weighted avg']['recall'],
                                       'weighted_avg_f1': report_dict['weighted avg']['f1-score'],

                                       'macro_avg_precision': report_dict['macro avg']['precision'],
                                       'macro_avg_recall': report_dict['macro avg']['recall'],
                                       'macro_avg_f1': report_dict['macro avg']['f1-score'],

                                       'top1_error': report_dict['top1_error'],
                                       'support': report_dict['macro avg']['support'],
                                       'best_epoch': report_dict['epoch']
                                       }])

    if 'df_general_report.csv' in os.listdir(os.getcwd()):
        df_general_report.to_csv('df_general_report.csv', index=False, header=False, mode='a')
    else:
        df_general_report.to_csv('df_general_report.csv', index=False, mode='a')


def test_model_external_data(model_name, dataset_name, external_dataset_name,
                             folder_to_evaluate, model_type):
    """
     Функция проведения оценки моделей по внешнему тестовому множеству данных
    :param model_name:      название модели
    :param dataset_name: наименование датасета
    :param external_dataset_name:  наименование внешнего датасета, по которому необходимо сделать тест модели
    :param folder_to_evaluate: папка с весами модели
    :param model_type: тип модели
    :return: None
    """

    assert dataset_name in config.DATASETS_LIST
    assert model_name in config.MODELS_LIST

    CONFIG = {'model': {'model_name': model_name,
                        'optimizer': config.MODEL_OPTIMIZER,
                        'pretrained': config.PRETRAINED,
                        'freeze_conv': config.FREEZE_CONV,
                        'epochs': config.EPOCHS,
                        'batch_size': config.BATCH_SIZE,
                        },

              'data': {'root_dir': os.getcwd().replace('src', 'datasets'),
                       'dataset_name': dataset_name,
                       'resize_img': config.RESIZE_IMG,
                       'num_workers': config.NUM_WORKERS,
                       'pytorch_aug': config.PYTORCH_AUG,
                       'save_aspect_ratio': config.SAVE_ASPECT_RATIO,
                       'imgaug_aug': config.IMGAUG_AUG,
                       'img_normalize': config.IMG_NORMALIZE,
                       },
              }

    DEVICE = get_computing_device()
    CONFIG.update({'seed': config.SEED})
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    FULL_DATASET_DIR = os.path.join(CONFIG['data']['root_dir'],
                                    CONFIG['data']['dataset_name'])
    CLASS_DICT = image_utils.make_class_dict(os.path.join(FULL_DATASET_DIR, 'df_img_meta.csv'))

    FULL_EXTERNAL_DATASET_DIR = os.path.join(CONFIG['data']['root_dir'],
                                             external_dataset_name)
    EXTERNAL_CLASS_DICT = image_utils.make_class_dict(os.path.join(FULL_EXTERNAL_DATASET_DIR, 'df_img_meta.csv'))

    if CONFIG['data']['num_workers'] > 0:
        # mp.set_start_method('spawn', force=True)
        cv2.setNumThreads(0)

    # Setting image transformations
    _pytorch_aug = CONFIG['data']['pytorch_aug']
    _img_normalize = CONFIG['data']['img_normalize']
    _save_aspect_ratio = CONFIG['data']['save_aspect_ratio']

    default_transforms_list = [augmentation.ImgAugTransform(with_aug=False),
                               image_utils.CustomResize((CONFIG['data']['resize_img'],
                                                         CONFIG['data']['resize_img']),
                                                        _save_aspect_ratio
                                                        ),
                               torchvision.transforms.ToTensor()
                               ]

    if _pytorch_aug:
        # Pytorch default & augmentation transform
        transform_list = [augmentation.ImgAugTransform(with_aug=CONFIG['data']['imgaug_aug']),
                          image_utils.CustomResize((CONFIG['data']['resize_img'],
                                                    CONFIG['data']['resize_img']),
                                                   _save_aspect_ratio
                                                   ),
                          torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
                          torchvision.transforms.RandomHorizontalFlip(),
                          torchvision.transforms.RandomRotation(20, resample=Image.BILINEAR),
                          torchvision.transforms.ToTensor()
                          ]
    else:
        transform_list = [augmentation.ImgAugTransform(with_aug=CONFIG['data']['imgaug_aug']),
                          image_utils.CustomResize((CONFIG['data']['resize_img'],
                                                    CONFIG['data']['resize_img']),
                                                   _save_aspect_ratio
                                                   ),
                          torchvision.transforms.ToTensor()
                          ]

    if _img_normalize:
        # Parameters were taken from Pytorch example for Imagenet
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L197-L198
        default_transforms_list.append(torchvision.transforms.Normalize(mean=config.IMG_NORMALIZE_MEAN,
                                                                        std=config.IMG_NORMALIZE_STD)
                                       )
    train_transforms = torchvision.transforms.Compose(transform_list)
    val_trainsforms = torchvision.transforms.Compose(default_transforms_list)

    # Set dictionaries for training, validation and testing
    TRAIN_DATASET_KWARGS = {'class_dict': CLASS_DICT,
                            'dataset_rootdir': FULL_DATASET_DIR,
                            'val_size': config.TESTVAL_SIZE,
                            'test_size': config.TEST_SIZE_FROM_TESTVAL,  # Доля от val_size
                            'seed': config.SEED,
                            'torch_transform': train_transforms,
                            }

    TESTVAL_DATASET_KWARGS = {'class_dict': EXTERNAL_CLASS_DICT,
                              'dataset_rootdir': FULL_EXTERNAL_DATASET_DIR,
                              'val_size': 1,
                              'test_size': 1,  # Доля от val_size
                              'seed': config.SEED,
                              'torch_transform': val_trainsforms,
                              }

    train_dataset = PytorchDataset('train', TRAIN_DATASET_KWARGS)
    test_dataset = PytorchDataset('test', TESTVAL_DATASET_KWARGS)
    test_data_loader = DataLoader(test_dataset, batch_size=CONFIG['model']['batch_size'],
                                  shuffle=True, num_workers=CONFIG['data']['num_workers'])
    # Число классов - относительно даных, на которых происходило обучение
    n_classes = len(CLASS_DICT.keys())

    if 'mobilenet' in CONFIG['model']['model_name']:
        nnet = mobilenet_selector.get_mobilenet(version=CONFIG['model']['model_name'],
                                                class_number=n_classes,
                                                pretrained=CONFIG['model']['pretrained'],
                                                freeze_conv=CONFIG['model']['freeze_conv'])
    else:
        nnet = resnet_selector.get_resnet(version=CONFIG['model']['model_name'],
                                          class_number=n_classes,
                                          pretrained=CONFIG['model']['pretrained'],
                                          freeze_conv=CONFIG['model']['freeze_conv'])

    # Train dataset img hashes to dataset hash. Universal for every machine.
    train_data_hash = \
        hashlib.sha1('_'.join(sorted(train_dataset.df_metadata['sha1'].values.tolist())).encode()).hexdigest()

    weights_folder = os.path.join(os.getcwd(), 'weights',
                                  f"""pytorch_{CONFIG['model']['model_name']}_{train_data_hash}""")
    weights_folder = os.path.join('/', *(weights_folder.split('/')[:-1] + [folder_to_evaluate]))

    EVAL_DICT = {'device': DEVICE,
                 'weights_folder': weights_folder,
                 'test_loader': test_data_loader,
                 'model': nnet,
                 'criterion': torch.nn.CrossEntropyLoss(),
                 'external_data_class_dict': EXTERNAL_CLASS_DICT,
                 'trained_data_class_dict': CLASS_DICT,
                 'data_hash': train_data_hash,
                 }
    reports_list = model_evaluate(EVAL_DICT)

    # Формирование табличного отчета по качеству классфикации
    # Для объективной оценки испольузем только лучшую эпоху по валидации.
    assert len(reports_list) == 1
    report_dict = reports_list[0]
    df_detailed_report = pd.DataFrame([])
    report_keys = list(report_dict.keys())
    for m in ['top1_error', 'accuracy', 'macro avg', 'weighted avg', 'epoch']:
        report_keys.remove(m)

    for class_name in report_keys:
        df_detailed_report = df_detailed_report.append([{'dataset_id': train_data_hash,
                                                         'task_name': dataset_name,
                                                         'class_imbalanced_tool': model_type,
                                                         'model_name': model_name,
                                                         'class': class_name,
                                                         'precision': report_dict[class_name]['precision'],
                                                         'recall': report_dict[class_name]['recall'],
                                                         'f1': report_dict[class_name]['f1-score'],
                                                         'support': report_dict[class_name]['support'],
                                                         'best_epoch': report_dict['epoch']
                                                         }
                                                        ]
                                                       )

    if 'df_detailed_report_external_data.csv' in os.listdir(os.getcwd()):
        df_detailed_report.to_csv('df_detailed_report_external_data.csv', index=False, header=False, mode='a')
    else:
        df_detailed_report.to_csv('df_detailed_report_external_data.csv', index=False, mode='a')

    df_general_report = pd.DataFrame([{'dataset_id': train_data_hash,
                                       'task_name': dataset_name,
                                       'class_imbalanced_tool': model_type,
                                       'model_name': model_name,

                                       'weighted_avg_precision': report_dict['weighted avg']['precision'],
                                       'weighted_avg_recall': report_dict['weighted avg']['recall'],
                                       'weighted_avg_f1': report_dict['weighted avg']['f1-score'],

                                       'macro_avg_precision': report_dict['macro avg']['precision'],
                                       'macro_avg_recall': report_dict['macro avg']['recall'],
                                       'macro_avg_f1': report_dict['macro avg']['f1-score'],

                                       'top1_error': report_dict['top1_error'],
                                       'support': report_dict['macro avg']['support'],
                                       'best_epoch': report_dict['epoch']
                                       }])

    if 'df_general_report_external_data.csv' in os.listdir(os.getcwd()):
        df_general_report.to_csv('df_general_report_external_data.csv', index=False, header=False, mode='a')
    else:
        df_general_report.to_csv('df_general_report_external_data.csv', index=False, mode='a')

    INVERT_CLASS_DICT = {v: k for k, v in CLASS_DICT.items()}
    y_true = []
    y_pred = []
    proba_list = []
    img_sha1 = []

    for i, test_batch in enumerate(test_data_loader):
        with torch.no_grad():
            inputs, labels = test_batch['image'], test_batch['class']
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            # Forward pass
            outputs = nnet(inputs)
            outputs_proba = F.softmax(outputs, dim=1)
            max_proba, indicies = torch.max(outputs_proba, 1)

            y_true.extend([INVERT_CLASS_DICT[x] for x in labels.cpu().numpy().tolist()])
            y_pred.extend([INVERT_CLASS_DICT[x] for x in indicies.cpu().numpy().tolist()])
            img_sha1.extend(test_batch['img_sha1'])
            proba_list.extend(max_proba.cpu().numpy().tolist())

    df_predictions = pd.DataFrame({'sha1': img_sha1, 'y_true': y_true, 'y_pred': y_pred, 'probability': proba_list})
    df_predictions.to_csv(f'../df_test_predictions_{external_dataset_name}_{model_name}_{model_type}.csv', index=False)
    logger.info(f'Saved: df_test_predictions_{external_dataset_name}_{model_name}_{model_type}.csv')


def make_interpretable_plots(model_name, dataset_name, folder_to_evaluate, model_type, manual_images=[]):
    """
    make_interpretable_plots
    :param model_name:
    :param dataset_name:
    :param folder_to_evaluate:
    :param model_type:
    :return:
    """

    assert dataset_name in config.DATASETS_LIST
    assert model_name in config.MODELS_LIST

    CONFIG = {'model': {'model_name': model_name,
                        'optimizer': config.MODEL_OPTIMIZER,
                        'pretrained': config.PRETRAINED,
                        'freeze_conv': config.FREEZE_CONV,
                        'epochs': config.EPOCHS,
                        'batch_size': config.BATCH_SIZE,
                        },

              'data': {'root_dir': os.getcwd().replace('src', 'datasets'),
                       'dataset_name': dataset_name,
                       'resize_img': config.RESIZE_IMG,
                       'num_workers': config.NUM_WORKERS,
                       'pytorch_aug': config.PYTORCH_AUG,
                       'save_aspect_ratio': config.SAVE_ASPECT_RATIO,
                       'imgaug_aug': config.IMGAUG_AUG,
                       'img_normalize': config.IMG_NORMALIZE,
                       },
              }

    DEVICE = get_computing_device()
    CONFIG.update({'seed': config.SEED})
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    FULL_DATASET_DIR = os.path.join(CONFIG['data']['root_dir'],
                                    CONFIG['data']['dataset_name'])
    CLASS_DICT = image_utils.make_class_dict(os.path.join(FULL_DATASET_DIR, 'df_img_meta.csv'))

    if CONFIG['data']['num_workers'] > 0:
        cv2.setNumThreads(0)

    # Setting image transformations
    _pytorch_aug = CONFIG['data']['pytorch_aug']
    _img_normalize = CONFIG['data']['img_normalize']
    _save_aspect_ratio = CONFIG['data']['save_aspect_ratio']

    default_transforms_list = [augmentation.ImgAugTransform(with_aug=False),
                               image_utils.CustomResize((CONFIG['data']['resize_img'],
                                                         CONFIG['data']['resize_img']),
                                                        _save_aspect_ratio
                                                        ),
                               torchvision.transforms.ToTensor()
                               ]

    if _pytorch_aug:
        # Pytorch default & augmentation transform
        transform_list = [augmentation.ImgAugTransform(with_aug=CONFIG['data']['imgaug_aug']),
                          image_utils.CustomResize((CONFIG['data']['resize_img'],
                                                    CONFIG['data']['resize_img']),
                                                   _save_aspect_ratio
                                                   ),
                          torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
                          torchvision.transforms.RandomHorizontalFlip(),
                          torchvision.transforms.RandomRotation(20, resample=Image.BILINEAR),
                          torchvision.transforms.ToTensor()
                          ]
    else:
        transform_list = [augmentation.ImgAugTransform(with_aug=CONFIG['data']['imgaug_aug']),
                          image_utils.CustomResize((CONFIG['data']['resize_img'],
                                                    CONFIG['data']['resize_img']),
                                                   _save_aspect_ratio
                                                   ),
                          torchvision.transforms.ToTensor()
                          ]

    if _img_normalize:
        # Parameters were taken from Pytorch example for Imagenet
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L197-L198
        default_transforms_list.append(torchvision.transforms.Normalize(mean=config.IMG_NORMALIZE_MEAN,
                                                                        std=config.IMG_NORMALIZE_STD)
                                       )
    train_transforms = torchvision.transforms.Compose(transform_list)
    val_transforms = torchvision.transforms.Compose(default_transforms_list)

    # Set dictionaries for training, validation and testing
    TRAIN_DATASET_KWARGS = {'class_dict': CLASS_DICT,
                            'dataset_rootdir': FULL_DATASET_DIR,
                            'val_size': config.TESTVAL_SIZE,
                            'test_size': config.TEST_SIZE_FROM_TESTVAL,  # Доля от val_size
                            'seed': config.SEED,
                            'torch_transform': train_transforms,
                            }

    TESTVAL_DATASET_KWARGS = {'class_dict': CLASS_DICT,
                              'dataset_rootdir': FULL_DATASET_DIR,
                              'val_size': config.TESTVAL_SIZE,
                              'test_size': config.TEST_SIZE_FROM_TESTVAL,  # Доля от val_size
                              'seed': config.SEED,
                              'torch_transform': val_transforms,
                              }

    models_last_layers = {'mobilenet_v2': 18,
                          'mobilenet_v3_large': 15,
                          'resnet18': 'layer4',
                          'resnet34': 'layer4',
                          'resnet50': 'layer4',
                          'resnet101': 'layer4',
                          'resnet152': 'layer4',
                          'wide_resnet50_2': 'layer4',
                          'wide_resnet101_2': 'layer4',
                          'resnext50_32x4d': 'layer4',
                          'resnext101_32x8d': 'layer4',

                          }

    train_dataset = PytorchDataset('train', TRAIN_DATASET_KWARGS)
    test_dataset = PytorchDataset('test', TESTVAL_DATASET_KWARGS)
    test_data_loader = DataLoader(test_dataset, batch_size=CONFIG['model']['batch_size'],
                                  shuffle=True, num_workers=CONFIG['data']['num_workers'])
    # Число классов - относительно даных, на которых происходило обучение
    n_classes = len(CLASS_DICT.keys())

    if 'mobilenet' in CONFIG['model']['model_name']:
        nnet = mobilenet_selector.get_mobilenet(version=CONFIG['model']['model_name'],
                                                class_number=n_classes,
                                                pretrained=CONFIG['model']['pretrained'],
                                                freeze_conv=CONFIG['model']['freeze_conv'])
    else:
        nnet = resnet_selector.get_resnet(version=CONFIG['model']['model_name'],
                                          class_number=n_classes,
                                          pretrained=CONFIG['model']['pretrained'],
                                          freeze_conv=CONFIG['model']['freeze_conv'])

    # Train dataset img hashes to dataset hash. Universal for every machine.
    train_data_hash = \
        hashlib.sha1('_'.join(sorted(train_dataset.df_metadata['sha1'].values.tolist())).encode()).hexdigest()

    weights_folder = os.path.join(os.getcwd(), 'weights',
                                  f"""pytorch_{CONFIG['model']['model_name']}_{train_data_hash}""")
    weights_folder = os.path.join('/', *(weights_folder.split('/')[:-1] + [folder_to_evaluate]))

    weights_list = [x for x in os.listdir(weights_folder) if '.pth' in x]
    # Проверим, что веса в директории одни
    assert len(weights_list) == 1
    weights_path = os.path.join(weights_folder, weights_list[0])

    # Отбор отдельных изображений
    assert type(manual_images) is list
    if manual_images:
        test_dataset.df_current_dataset = test_dataset.df_metadata[test_dataset.df_metadata['sha1'].isin(manual_images)]

    lime_container = model_explain.LimeExplainContainer(model=nnet,
                                                        resize=CONFIG['data']['resize_img'],
                                                        meta_dataset=test_dataset.df_current_dataset,
                                                        class_dict=CLASS_DICT,
                                                        weights_path=weights_path,
                                                        device=DEVICE)
    lime_container.generate_test_explanations(top_labels=config.LIME_TOP_LABELS,
                                              num_features=config.LIME_NUM_FEATURES,
                                              num_samples=config.LIME_NUM_SAMPLES,
                                              save_img=config.LIME_SAVE_IMG,
                                              folder_prefix=model_type
                                              )

    gradcam_container = model_explain.GradCamContainer(model=nnet,
                                                       model_name=model_name,
                                                       resize=CONFIG['data']['resize_img'],
                                                       meta_dataset=test_dataset.df_current_dataset,
                                                       class_dict=CLASS_DICT,
                                                       weights_path=weights_path,
                                                       device=DEVICE)
    gradcam_container.generate_heatmaps(layer_min=models_last_layers[model_name],
                                        layer_max=models_last_layers[model_name],
                                        folder_prefix=model_type
                                        )

    rise_container = model_explain.RiseContainer(model=nnet,
                                                 class_dict=CLASS_DICT,
                                                 meta_dataset=test_dataset.df_current_dataset,
                                                 input_size=(CONFIG['data']['resize_img'],
                                                             CONFIG['data']['resize_img']),
                                                 weights_path=weights_path,
                                                 device=DEVICE,
                                                 gpu_batch=config.RISE_GPU_BATCH)
    rise_container.generate_rise_heatmaps(folder_prefix=model_type)


def get_detailed_predictions(model_name, dataset_name, folder_to_evaluate, model_type):

    logger.info('\nGetting detailed test predictions\n')

    assert dataset_name in config.DATASETS_LIST
    assert model_name in config.MODELS_LIST

    CONFIG = {'model': {'model_name': model_name,
                        'optimizer': config.MODEL_OPTIMIZER,
                        'pretrained': config.PRETRAINED,
                        'freeze_conv': config.FREEZE_CONV,
                        'epochs': config.EPOCHS,
                        'batch_size': config.BATCH_SIZE,
                        },

              'data': {'root_dir': os.getcwd().replace('src', 'datasets'),
                       'dataset_name': dataset_name,
                       'resize_img': config.RESIZE_IMG,
                       'num_workers': config.NUM_WORKERS,
                       'pytorch_aug': config.PYTORCH_AUG,
                       'save_aspect_ratio': config.SAVE_ASPECT_RATIO,
                       'imgaug_aug': config.IMGAUG_AUG,
                       'img_normalize': config.IMG_NORMALIZE,
                       },
              }

    DEVICE = get_computing_device()
    CONFIG.update({'seed': config.SEED})
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    FULL_DATASET_DIR = os.path.join(CONFIG['data']['root_dir'],
                                    CONFIG['data']['dataset_name'])
    CLASS_DICT = image_utils.make_class_dict(os.path.join(FULL_DATASET_DIR, 'df_img_meta.csv'))
    INVERT_CLASS_DICT = {v: k for k, v in CLASS_DICT.items()}

    if CONFIG['data']['num_workers'] > 0:
        cv2.setNumThreads(0)

    # Setting image transformations
    _pytorch_aug = CONFIG['data']['pytorch_aug']
    _img_normalize = CONFIG['data']['img_normalize']
    _save_aspect_ratio = CONFIG['data']['save_aspect_ratio']

    default_transforms_list = [augmentation.ImgAugTransform(with_aug=False),
                               image_utils.CustomResize((CONFIG['data']['resize_img'],
                                                         CONFIG['data']['resize_img']),
                                                        _save_aspect_ratio
                                                        ),
                               torchvision.transforms.ToTensor()
                               ]

    if _pytorch_aug:
        # Pytorch default & augmentation transform
        transform_list = [augmentation.ImgAugTransform(with_aug=CONFIG['data']['imgaug_aug']),
                          image_utils.CustomResize((CONFIG['data']['resize_img'],
                                                    CONFIG['data']['resize_img']),
                                                   _save_aspect_ratio
                                                   ),
                          torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
                          torchvision.transforms.RandomHorizontalFlip(),
                          torchvision.transforms.RandomRotation(20, resample=Image.BILINEAR),
                          torchvision.transforms.ToTensor()
                          ]
    else:
        transform_list = [augmentation.ImgAugTransform(with_aug=CONFIG['data']['imgaug_aug']),
                          image_utils.CustomResize((CONFIG['data']['resize_img'],
                                                    CONFIG['data']['resize_img']),
                                                   _save_aspect_ratio
                                                   ),
                          torchvision.transforms.ToTensor()
                          ]

    if _img_normalize:
        # Parameters were taken from Pytorch example for Imagenet
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L197-L198
        default_transforms_list.append(torchvision.transforms.Normalize(mean=config.IMG_NORMALIZE_MEAN,
                                                                        std=config.IMG_NORMALIZE_STD)
                                       )
    train_transforms = torchvision.transforms.Compose(transform_list)
    val_trainsforms = torchvision.transforms.Compose(default_transforms_list)

    # Set dictionaries for training, validation and testing
    TRAIN_DATASET_KWARGS = {'class_dict': CLASS_DICT,
                            'dataset_rootdir': FULL_DATASET_DIR,
                            'val_size': config.TESTVAL_SIZE,
                            'test_size': config.TEST_SIZE_FROM_TESTVAL,  # Доля от val_size
                            'seed': config.SEED,
                            'torch_transform': train_transforms,
                            }

    TESTVAL_DATASET_KWARGS = {'class_dict': CLASS_DICT,
                              'dataset_rootdir': FULL_DATASET_DIR,
                              'val_size': config.TESTVAL_SIZE,
                              'test_size': config.TEST_SIZE_FROM_TESTVAL,  # Доля от val_size
                              'seed': config.SEED,
                              'torch_transform': val_trainsforms,
                              }

    train_dataset = PytorchDataset('train', TRAIN_DATASET_KWARGS)
    test_dataset = PytorchDataset('test', TESTVAL_DATASET_KWARGS)
    test_data_loader = DataLoader(test_dataset, batch_size=CONFIG['model']['batch_size'],
                                  shuffle=True, num_workers=CONFIG['data']['num_workers'])
    # Число классов - относительно даных, на которых происходило обучение
    n_classes = len(CLASS_DICT.keys())

    if 'mobilenet' in CONFIG['model']['model_name']:
        nnet = mobilenet_selector.get_mobilenet(version=CONFIG['model']['model_name'],
                                                class_number=n_classes,
                                                pretrained=CONFIG['model']['pretrained'],
                                                freeze_conv=CONFIG['model']['freeze_conv'])
    else:
        nnet = resnet_selector.get_resnet(version=CONFIG['model']['model_name'],
                                          class_number=n_classes,
                                          pretrained=CONFIG['model']['pretrained'],
                                          freeze_conv=CONFIG['model']['freeze_conv'])

    # Train dataset img hashes to dataset hash. Universal for every machine.
    train_data_hash = \
        hashlib.sha1('_'.join(sorted(train_dataset.df_metadata['sha1'].values.tolist())).encode()).hexdigest()
    weights_folder = os.path.join(os.getcwd(), 'weights',
                                  f"""pytorch_{CONFIG['model']['model_name']}_{train_data_hash}""")
    weights_folder = os.path.join('/', *(weights_folder.split('/')[:-1] + [folder_to_evaluate]))
    weights_list = [x for x in os.listdir(weights_folder) if '.pth' in x]

    # Проверим, что веса в директории  одни
    assert len(weights_list) == 1
    weights_path = os.path.join(weights_folder, weights_list[0])

    nnet.load_state_dict(torch.load(weights_path))
    nnet.eval()
    nnet.to(DEVICE)

    y_true = []
    y_pred = []
    proba_list = []
    img_sha1 = []

    for i, test_batch in enumerate(test_data_loader):
        with torch.no_grad():
            inputs, labels = test_batch['image'], test_batch['class']
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            # Forward pass
            outputs = nnet(inputs)
            outputs_proba = F.softmax(outputs, dim=1)
            max_proba, indicies = torch.max(outputs_proba, 1)

            y_true.extend([INVERT_CLASS_DICT[x] for x in labels.cpu().numpy().tolist()])
            y_pred.extend([INVERT_CLASS_DICT[x] for x in indicies.cpu().numpy().tolist()])
            img_sha1.extend(test_batch['img_sha1'])
            proba_list.extend(max_proba.cpu().numpy().tolist())

    df_predictions = pd.DataFrame({'sha1': img_sha1, 'y_true': y_true, 'y_pred': y_pred, 'probability': proba_list})
    df_predictions.to_csv(f'../df_test_predictions_{dataset_name}_{model_name}_{model_type}.csv', index=False)
    logger.info(f'Saved: df_test_predictions_{dataset_name}_{model_name}_{model_type}.csv')


def print_model_structure(model_name):

    """
    Print model pytorch structure in logger debugging
    :param model_name:
    :return:
    """

    CONFIG = {'model': {'model_name': model_name,
                        'pretrained': config.PRETRAINED,
                        'freeze_conv': config.FREEZE_CONV,
                        },
              }

    if 'mobilenet' in CONFIG['model']['model_name']:
        nnet = mobilenet_selector.get_mobilenet(version=CONFIG['model']['model_name'],
                                                class_number=1,
                                                pretrained=CONFIG['model']['pretrained'],
                                                freeze_conv=CONFIG['model']['freeze_conv'])
    else:
        nnet = resnet_selector.get_resnet(version=CONFIG['model']['model_name'],
                                          class_number=1,
                                          pretrained=CONFIG['model']['pretrained'],
                                          freeze_conv=CONFIG['model']['freeze_conv'])
    logger.info(nnet)


if __name__ == '__main__':

    print_model_structure(model_name=config.TESTING_MODEL_NAME)

    if config.TESTING_ON_TEST_PART_OF_GENERAL_DATA:
        for _type, folder_name in config.TESTING_MODEL_WEIGHTS.items():
            logger.info(f'Model type: {_type.upper()}')
            test_model(model_name=config.TESTING_MODEL_NAME, dataset_name=config.TESTING_DATASET_NAME,
                       folder_to_evaluate=folder_name, model_type=_type)

    if config.TESTING_ON_EXTERNAL_DATA:
        for _type, folder_name in config.TESTING_MODEL_WEIGHTS.items():
            logger.info(f'Model type: {_type.upper()}')
            test_model_external_data(model_name=config.TESTING_MODEL_NAME,
                                     dataset_name=config.TESTING_DATASET_NAME,
                                     external_dataset_name=config.TESTING_EXTERNAL_DATASET_NAME,
                                     folder_to_evaluate=folder_name, model_type=_type)

    if config.TESTING_INTERPRETABLE_PLOTS:
        for _type, folder_name in config.TESTING_MODEL_WEIGHTS.items():
            logger.info(f'Model type: {_type.upper()}')
            make_interpretable_plots(model_name=config.TESTING_MODEL_NAME, dataset_name=config.TESTING_DATASET_NAME,
                                     folder_to_evaluate=folder_name, model_type=_type)

    if config.TESTING_DETAILED_TEST_PREDS:
        for _type, folder_name in config.TESTING_MODEL_WEIGHTS.items():
            logger.info(f'Model type: {_type.upper()}')
            get_detailed_predictions(model_name=config.TESTING_MODEL_NAME, dataset_name=config.TESTING_DATASET_NAME,
                                     folder_to_evaluate=folder_name, model_type=_type)

