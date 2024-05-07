# Model Parameter
MODEL_SAVE = True
MODEL_NAME = "Model9v3_whitebq_aug_onlydense_transfer"
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 128

# Directory Parameter
MODEL_DIR_NAME = "models/keras"
TEST_RESULT_DIR_NAME = "test_results"
# Training Parameter
SAVE_HISTORY = True
EPOCHS = 200
BATCH_SIZE = 24
TF_SEED = 42
LEARNING_RATE = 0.001
PATIENCE = 5

# Data Parameter
IAM_DATASET_PATH = "dataset/iam_dataset"
TRANSFER_DATASET_PATH = "dataset/transfer_dataset"