
import os
import cv2
import torch
import torchvision

import hashlib
import logging
import numpy as np

from PIL import Image
from torch.utils.data import DataLoader

from pandas.io.json._normalize import nested_to_record

from utils import nn_utils, image_utils, augmentation
from utils.model_training_utils import get_computing_device, load_pytorch_model, get_optimizer, \
                                       PytorchDataset, SaveBestModelCallback, model_train

import resnet_selector
import mobilenet_selector

import neptune_logger
import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )


def make_experiment(model_name, dataset_name, imbalanced_tool):

    """
    Функция запуска экспериментов по обучению моделей
    :param model_name:       наименование модели
    :param dataset_name:     наименование датасета
    :param imbalanced_tool:  наименование подхода борьбы с несбалансированностью классов
    :return: None
    """

    assert imbalanced_tool in config.IMBALANCED_TOOL_LIST
    assert dataset_name in config.DATASETS_LIST
    assert model_name in config.MODELS_LIST

    # Dict for imbalanced params setting
    imbalanced_dict = {'train_sampler': {'default': False,
                                         'weighted_loss': False,
                                         'train_sampler': True
                                         },
                       'weighted_loss': {'default': False,
                                         'weighted_loss': True,
                                         'train_sampler': False
                                         },
                       'tag': {'default': 'default',
                               'weighted_loss': 'weighted_loss',
                               'train_sampler': 'custom_batch_sampler'
                               }
                       }

    # General config
    """
    ResNets:'resnet18','resnet34', 'resnet50', 'resnet101', 'resnet152',
            'wide_resnet50_2', 'wide_resnet101_2', 'resnext50_32x4d', 'resnext101_32x8d'
    MobileNets: 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small'

    mobilenet_v3_small пока отложить
    """
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

              'imbalanced_tools': {'train_sampler': imbalanced_dict['train_sampler'][imbalanced_tool],
                                   'weighted_loss': imbalanced_dict['weighted_loss'][imbalanced_tool],
                                   }
              }

    NEPTUNE_EXPERIMENT_TAG_LIST = [CONFIG['data']['dataset_name'],
                                   CONFIG['model']['model_name'],
                                   'imbalanced_tools',
                                   imbalanced_dict['tag'][imbalanced_tool],
                                   ]

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
        transform_list.append(torchvision.transforms.Normalize(mean=config.IMG_NORMALIZE_MEAN,
                                                               std=config.IMG_NORMALIZE_STD)
                              )
        default_transforms_list.append(torchvision.transforms.Normalize(mean=config.IMG_NORMALIZE_MEAN,
                                                                        std=config.IMG_NORMALIZE_STD)
                                       )

    train_transforms = torchvision.transforms.Compose(transform_list)
    val_trainsforms = torchvision.transforms.Compose(default_transforms_list)

    # Set dictionaries for training, validation and testing
    TRAIN_DATASET_KWARGS = {'class_dict': CLASS_DICT,
                            'dataset_rootdir': FULL_DATASET_DIR,
                            'val_size': config.TESTVAL_SIZE,
                            'test_size': config.TEST_SIZE_FROM_TESTVAL,   # Доля от val_size
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
    val_dataset = PytorchDataset('val', TESTVAL_DATASET_KWARGS)
    test_dataset = PytorchDataset('test', TESTVAL_DATASET_KWARGS)

    # Set data loaders
    # Info: наблюдается deadlock в Jupyter/ JupyterLab со сторонней аугментацией при задании workers
    # В Pycharm (.py file) deadlock не наблюдается, можно ставить для ускорения.

    # What sampler for batching to choose
    if CONFIG['imbalanced_tools']['train_sampler']:

        _train_shuffle = False
        _train_sampler = nn_utils.ImbalancedDatasetSampler(dataset=train_dataset,
                                                           class_dict=CLASS_DICT,
                                                           random_state=config.SEED)
    else:
        _train_shuffle = True
        _train_sampler = None

    logger.debug(f"""Configuring data loaders. Using cpus for multiprocessing: {CONFIG['data']['num_workers']}.""")
    train_data_loader = DataLoader(train_dataset, batch_size=CONFIG['model']['batch_size'],
                                   shuffle=_train_shuffle, sampler=_train_sampler,
                                   num_workers=CONFIG['data']['num_workers'])
    val_data_loader = DataLoader(val_dataset, batch_size=CONFIG['model']['batch_size'],
                                 shuffle=True, num_workers=CONFIG['data']['num_workers'])
    # test_data_loader = DataLoader(test_dataset, batch_size=CONFIG['model']['batch_size'],
    #                               shuffle=True, num_workers=CONFIG['data']['num_workers'])

    # Optimizers and their parameters
    # https://pytorch.org/docs/stable/optim.html
    OPTIM_KWARGS = {'optimizer': {'adadelta': {'lr': 1.0, 'rho': 0.9, 'eps': 1e-06, 'weight_decay': 0},

                                  'adagrad': {'lr': 0.01, 'lr_decay': 0, 'weight_decay': 0,
                                              'initial_accumulator_value': 0, 'eps': 1e-10},

                                  'adam': {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08,
                                           'weight_decay': 0, 'amsgrad': False},

                                  'adamw': {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08,

                                            'weight_decay': 0.01, 'amsgrad': False},

                                  'adamax': {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08,
                                             'weight_decay': 0},

                                  'rms_prop': {'lr': 0.01, 'alpha': 0.99, 'eps': 1e-08, 'weight_decay': 0,
                                               'momentum': 0, 'centered': False}
                                  },

                    'lr_scheduler': {'mode': 'min', 'factor': 0.95, 'patience': 10, 'verbose': False,
                                     'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 0, 'min_lr': 1e-9,
                                     'eps': 1e-08}
                    }

    # Select and configure model to be trained
    # Number of classes
    n_classes = len(TRAIN_DATASET_KWARGS['class_dict'].keys())

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

    # Whether to use weighted loss or not
    if CONFIG['imbalanced_tools']['weighted_loss']:
        _label_to_count = train_dataset.df_metadata['class'].value_counts().to_dict()
        _unnorm_weights = {k: 1 / v for k, v in _label_to_count.items()}
        _normed_weights = {k: v / sum(_unnorm_weights.values()) for k, v in _unnorm_weights.items()}

        _loss_weight = [v for k, v in sorted(_normed_weights.items(), key=lambda x: x[0])]
        _loss_weight = torch.FloatTensor(_loss_weight).to(DEVICE)

    else:
        _loss_weight = None

    # Set training keyword argumetns
    criterion = torch.nn.CrossEntropyLoss(weight=_loss_weight)
    optimizer = get_optimizer(CONFIG['model']['optimizer'],
                              nnet,
                              OPTIM_KWARGS['optimizer'])

    # Train dataset img hashes to dataset hash. Universal for every machine.
    train_data_hash = hashlib.sha1(
        '_'.join(sorted(train_dataset.df_metadata['sha1'].values.tolist())).encode()).hexdigest()

    training_kwargs = {'device': DEVICE,

                       'criterion': criterion,
                       'optimizer': optimizer,
                       'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                                  **OPTIM_KWARGS['lr_scheduler']),

                       'data_loaders': {'train_loader': train_data_loader,
                                        'val_loader': val_data_loader},
                       'total_epochs': CONFIG['model']['epochs'],
                       'batch_size': CONFIG['model']['batch_size'],
                       'target_names': list(TRAIN_DATASET_KWARGS['class_dict'].keys()),

                       'model_name': f"""{CONFIG['model']['model_name']}_{train_data_hash}""",
                       'class_dict': CLASS_DICT,
                       'save_path': os.path.join(os.getcwd(), 'weights'),
                       'general_config': CONFIG
                       }

    logger.debug(f'Training kwargs'.center(50, '='))
    logger.debug(''.join(list(f'{k}: {v}\n' for k, v in training_kwargs.items())))

    weights_folder = os.path.join(os.getcwd(), 'weights',
                                  f"""pytorch_{CONFIG['model']['model_name']}_{train_data_hash}""")

    if config.USE_NEPTUNE:
        logger.debug('Neptune logger is ON.')
        # Сбор параметров для логгирования
        TOTAL_PARAMS = {'config': CONFIG,
                        'training_kwargs': training_kwargs,
                        'train_dataset_kwargs': TRAIN_DATASET_KWARGS,
                        'testval_dataset_kwargs': TESTVAL_DATASET_KWARGS,
                        'weights_folder': weights_folder,
                        'training_data_sha1': train_data_hash,
                        }
        neptune_kwargs = {'params': nested_to_record(TOTAL_PARAMS, sep='_'),
                          'artifact_path': os.path.join(os.getcwd().replace('src', 'log_artifacts'), 'artifacts'),
                          'image_path': os.path.join(os.getcwd().replace('src', 'log_artifacts'), 'images'),
                          'tag_list': NEPTUNE_EXPERIMENT_TAG_LIST,
                          'training_data_sha1': train_data_hash
                          }
        neptune_class = neptune_logger.NeptuneLogger(neptune_kwargs)
    else:
        logger.debug('Neptune logger is OFF.')
        neptune_class = None

    # Training
    model_train(nnet, training_kwargs, neptune_class=neptune_class)

    logger.debug(f'Done: model_name: {model_name}, dataset: {dataset_name}, imbalanced_tool: {imbalanced_tool}')


if __name__ == '__main__':

    """
    Arguments for training experiments

    ['default', 'weighted_loss', 'train_sampler']

     ResNets:
             'resnet18','resnet34', 'resnet50', 'resnet101', 'resnet152',
             'wide_resnet50_2', 'wide_resnet101_2', 'resnext50_32x4d', 'resnext101_32x8d'
     MobileNets: 
             'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small'
    """

    for tool in ['default', 'weighted_loss', 'train_sampler']:
        make_experiment(model_name='mobilenet_v2',
                        dataset_name='test_data',
                        imbalanced_tool=tool)

