

# Available datasets and lists should be placed here
DATASETS_LIST = ['14_08_20_adelphocoris_male_female_all_previous',
                 '14_08_20_adelphocoris_general_classification_all_previous',
                 '17_09_20_adelphocoris_and_outgroup',
                 '13_10_20_stenus_general_classification',
                 '12_11_20_stenus_general_classification',
                 'test_data'
                 ]
# List of possible models for training
MODELS_LIST = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
               'wide_resnet50_2', 'wide_resnet101_2', 'resnext50_32x4d', 'resnext101_32x8d',
               'mobilenet_v2', 'mobilenet_v3_large'
               ]
# List of possible tools for class-imbalance problem
IMBALANCED_TOOL_LIST = ['weighted_loss', 'train_sampler', 'default']

# Constants for model training
# General training params
PRETRAINED = True
FREEZE_CONV = False
NUM_WORKERS = 8
EPOCHS = 80
BATCH_SIZE = 16
# Image transformation
RESIZE_IMG = 300
PYTORCH_AUG = True
IMGAUG_AUG = True
SAVE_ASPECT_RATIO = False
IMG_NORMALIZE = True
IMG_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
IMG_NORMALIZE_STD = [0.229, 0.224, 0.225]
# Dataset split
TESTVAL_SIZE = 0.3
TEST_SIZE_FROM_TESTVAL = 0.5
# Model optimizer
MODEL_OPTIMIZER = 'adamax'
# Random seed
SEED = 42
# You can use Neptune as training logger
USE_NEPTUNE = False
NEPTUNE_PROJECT_QUAL_NAME = r''
NEPTUNE_TOKEN = ''

# Constants for model testing
TESTING_MODEL_NAME = 'mobilenet_v2'              # Model name: mobilenet, resnet, etc, see config.py
TESTING_DATASET_NAME = 'test_data'               # Dataset name
TESTING_EXTERNAL_DATASET_NAME = ''               # External data dataset to test
TESTING_ON_EXTERNAL_DATA = False                 # Test model on external dataset
TESTING_ON_TEST_PART_OF_GENERAL_DATA = True      # Test model on test subsample
TESTING_INTERPRETABLE_PLOTS = True               # Make interpretable plots on test data
TESTING_DETAILED_TEST_PREDS = True               # Save predictions by sample based on test data
TESTING_MODEL_WEIGHTS = {
    'default': '',
}

# Constants for model interpretability methods
# LIME
LIME_TOP_LABELS = 1
LIME_NUM_FEATURES = 5
LIME_NUM_SAMPLES = 2000
LIME_SAVE_IMG = True
# RISE
RISE_GPU_BATCH = 8

