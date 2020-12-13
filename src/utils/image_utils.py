
import os
import cv2
import tqdm
import time
import hashlib
import datetime
import numpy as np
import pandas as pd
import torchvision

from PIL import Image


def timeit(method):
    """
    Декоратор для замеров времени работы функций
    :param method:
    :return:
    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r:  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed


@timeit
def create_hashed_img_dataset(project_path: str, rel_raw_dataset_path: str, rel_registered_image_path: str,
                              meta_info_path: str
                              ) -> None:
    """
    Формирование датасета с изображениями с уникальным ключом - sha1 хешем изображения.
    Изображения берутся из raw_dataset_path, обсчитываются хеши, новые изображения с именем хеша
    помещаются в registered_image_path.

    Example: create_hashed_img_dataset(RAW_IMAGE_PATH, REGISTERED_IMAGE_PATH, 'bee')

    :param raw_dataset_path: str
    :param registered_image_path: str
    :return: None
    """
    print('Registering dataset.')
    print(f'Project path: {project_path}\nRelative raw data path: {rel_raw_dataset_path}')

    df_img_meta = pd.DataFrame(columns=['datetime', 'raw_dataset_path', 'reg_img_path', 'sha1', 'class'])
    _class_folders_dict = dict()

    for class_folder in os.listdir(os.path.join(project_path, rel_raw_dataset_path)):
        _class_folders_dict.update({class_folder: os.path.join(rel_raw_dataset_path, class_folder)})

    for class_name, class_folder_path in tqdm.tqdm(_class_folders_dict.items()):
        print('\nProcessing class: ', class_name)
        _image_list = [os.path.join(class_folder_path, x) for x in
                       os.listdir(os.path.join(project_path, class_folder_path))]

        existing_sha1 = []
        # Проверить наличие регистра данных и выгрузить хеши изображений в случае его наличия
        if 'df_img_meta.csv' in os.listdir(os.path.join(project_path, meta_info_path)):
            df_img_meta_history = pd.read_csv(os.path.join(project_path, meta_info_path, 'df_img_meta.csv'))
            existing_sha1 = df_img_meta_history['sha1'].values.tolist()
        #  Пройтись по всем изображениям конкретного класса
        for img_path in _image_list:

            check_data_format(img_path.split('.')[-1], ['jpg', 'jpeg', 'png'])

            img = cv2.imread(os.path.join(project_path, img_path))
            img_hash = hashlib.sha1(img.tostring()).hexdigest()

            # Проверить факт того, что изображение уже было обработано
            if img_hash in existing_sha1:
                print(f'Image: {img_hash} is already in meta_info. Skipping.')
                continue
            # Запись метаданных
            img_format = img_path.split('.')[-1]
            meta_info = {'datetime': str(datetime.datetime.now())[:19],
                         'raw_dataset_path': img_path,
                         'reg_img_path': os.path.join(rel_registered_image_path, f'{img_hash}.{img_format}'),
                         'sha1': img_hash,
                         'class': class_name.replace(' ', '_')
                         }
            df_img_meta = df_img_meta.append([meta_info])
            cv2.imwrite(os.path.join(project_path, rel_registered_image_path, f'{img_hash}.{img_format}'), img)

    if 'df_img_meta.csv' in os.listdir(os.path.join(project_path, meta_info_path)):
        df_img_meta.to_csv(os.path.join(project_path, meta_info_path, 'df_img_meta.csv'),
                           index=False, header=False, mode='a')
    else:
        df_img_meta.to_csv(os.path.join(project_path, meta_info_path, 'df_img_meta.csv'), index=False)


def rename_all_images(path: str, prefix: str, img_format='jpg') -> None:
    """
    Переименовать пул изображений с поомщью хеша и добавленного префикса
    :param path:       Директория с изображениями
    :param prefix:     Префикс для добавления
    :param img_format: Формат изображений
    :return: None
    """
    for root, dirs, files in os.walk(path):
        for i, f in tqdm.tqdm(enumerate(files)):

            absname = os.path.join(root, f)
            img = cv2.imread(absname)
            img_hash = hashlib.sha1(img.tostring()).hexdigest()
            newname = os.path.join(root, f'{img_hash}_{prefix}.{img_format}')
            os.rename(absname, newname)


def to_categorical(y, num_classes):
    """
    1-hot encodes a tensor
    :param y:           y variable
    :param num_classes: number of classes
    :return: 1-hot encoded y variable
    """
    return np.eye(num_classes, dtype='uint8')[y]


def make_class_dict(data_path):
    df = pd.read_csv(data_path)
    sorted_class_names = sorted(df['class'].unique().tolist())
    return dict([(j, i) for i, j in enumerate(sorted_class_names)])


def crop_image(img):
    """
    Обрезка изображения по определенным правилам
    :param img:
    :return:
    """
    pass


def check_data_format(data_format, types):
    if data_format.lower() not in types:
        raise ValueError(f'Data format is not recognized: {data_format}')


class CustomResize:

    def __init__(self, size, save_aspect_ratio=False):
        """
        Custom resize with aspect ratio saving option based on torchvision.transforms.Resize
        :param size:              tuple: (int, int) with order: (width, height)
        :param save_aspect_ratio: bool, whether or not to save aspect ratio or not
        """

        super(CustomResize, self).__init__()
        self.size = size
        self.save_aspect_ratio = save_aspect_ratio
        self.default_resize = torchvision.transforms.Resize(self.size)
        self.interpolation = self.default_resize.interpolation

    def __call__(self, img):

        if self.save_aspect_ratio:

            # Resize with respect to aspect_ratio = width//height
            width, height = img.size
            _asp_ratio = width / height
            new_width = self.size[0]
            new_height = int(new_width // _asp_ratio)

            # import torch.nn.functional as F
            new_img = torchvision.transforms.functional.resize(img, (new_width, new_height),
                                                               self.interpolation)
            # Pad resized image to size network input
            output = Image.new("RGB", self.size)
            output.paste(new_img,
                         ((self.size[0] - new_width) // 2,
                          (self.size[0] - new_height) // 2)
                         )
        else:
            output = self.default_resize(img)

        return output



