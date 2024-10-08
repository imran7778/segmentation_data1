TRAIN_DATA_PATH = '/content/segmentation_data/data128/'
NEW_TRAIN_DATA_PATH = '/content/segmentation_data/New_train_data/'
TEST_DATA_PATH = '/content/segmentation_data/Test_data/'
SUBMISSION_DATA_PATH = '/content/segmentation_data/Submission_data'
MODEL_CHECKPOINT_DIR = 'Checkpoints/'
WEIGHTS = 'Model_Weights.hdf5'
AUGMENT_TRAIN_DATA = False
CREATE_EXTRA_DATA = True
IMG_ROWS = 128
IMG_COLS = 128
IMG_START_NUM = 164
SMOOTH = 1.0
CLEAN_THRESH = 20
THRESH = 100
BATCH_SIZE = 3
EPOCHS = 50
BASE_LR = 1e-04
PATIENCE = 10