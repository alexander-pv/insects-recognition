
import os
import logging
import config
import prepare_dataset
import model_training
import models_test

if __name__ == '__main__':

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S'
                        )

    # Prepare dataset
    prepare_dataset.register_image_dataset(data_folder='12_11_20_stenus_general_classification',
                                           work_dir=os.getcwd()
                                           )
    # Train model
    for tool in ['default', 'weighted_loss', 'train_sampler']:
        model_training.make_experiment(model_name='mobilenet_v2',
                                       dataset_name='test_data',
                                       imbalanced_tool=tool)

    # Test model and make interpretability plots

    if config.TESTING_ON_TEST_PART_OF_GENERAL_DATA:
        for type, folder_name in config.TESTING_MODEL_WEIGHTS.items():
            logger.debug(f'Model type: {type.upper()}')
            models_test.test_model(model_name=config.TESTING_MODEL_NAME,
                                   dataset_name=config.TESTING_DATASET_NAME,
                                   folder_to_evaluate=folder_name, model_type=type)

    if config.TESTING_ON_EXTERNAL_DATA:
        for type, folder_name in config.TESTING_MODEL_WEIGHTS.items():
            logger.debug(f'Model type: {type.upper()}')
            models_test.test_model_external_data(model_name=config.TESTING_MODEL_NAME,
                                                 dataset_name=config.TESTING_DATASET_NAME,
                                                 external_dataset_name=config.TESTING_EXTERNAL_DATASET_NAME,
                                                 folder_to_evaluate=folder_name, model_type=type)

    if config.TESTING_INTERPRETABLE_PLOTS:
        for type, folder_name in config.TESTING_MODEL_WEIGHTS.items():
            logger.debug(f'Model type: {type.upper()}')
            models_test.make_interpretable_plots(model_name=config.TESTING_MODEL_NAME,
                                                 dataset_name=config.TESTING_DATASET_NAME,
                                                 folder_to_evaluate=folder_name, model_type=type)

    if config.TESTING_DETAILED_TEST_PREDS:
        for type, folder_name in config.TESTING_MODEL_WEIGHTS.items():
            logger.debug(f'Model type: {type.upper()}')
            models_test.get_detailed_predictions(model_name=config.TESTING_MODEL_NAME,
                                                 dataset_name=config.TESTING_DATASET_NAME,
                                                 folder_to_evaluate=folder_name, model_type=type)



