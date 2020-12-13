
import os
from utils import image_utils


def register_image_dataset(data_folder, work_dir):
    """
    Prepare dataset for training and other experiments.

    Please, place initial images to [work_dir/data_folder/raw] where each subfolder contains images of a certain class
    This function will create 'registered' folder near 'raw' and save mapping .csv table for prepared images.
    :param data_folder: folder name of a dataset
    :param work_dir: cwd from os library
    :return: None
    """
    # Название интересующей директории с изображениями
    # Формирование полного пути до интересующей директории и директории зарегистрированных изображений
    _project_wd = work_dir.replace(r'/src', '')
    _rel_raw_dataset_path = os.path.join('datasets', data_folder, 'raw')
    _rel_registered_image_path = os.path.join('datasets', data_folder, 'registered')
    # Создать директорию с зарегистрированными изображениями в случае её отсутствия
    if 'registered' not in os.listdir(work_dir.replace(r'/src', f'/datasets/{data_folder}')):
        os.mkdir(os.path.join(_project_wd, _rel_registered_image_path))
        # Путь сохранения метаданных по изображениям
        _meta_info_path = _rel_raw_dataset_path.replace('raw', '')
        # Регистрация изображений
        image_utils.create_hashed_img_dataset(_project_wd, _rel_raw_dataset_path,
                                              _rel_registered_image_path, _meta_info_path)
    else:
        print(f'Path {os.path.join(_project_wd, _rel_registered_image_path)} exists. Skipping.')
