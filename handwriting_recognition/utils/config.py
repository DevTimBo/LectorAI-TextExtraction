# Model Parameter
MODEL_SAVE = True
MODEL_NAME = "Model9v3_Words"
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 128

# Pay attention to HYPERPARAMETER_TUNE for file location
HYPERPARAMETER_TUNE = False

# Directory Parameter
MODEL_DIR_NAME = "models/keras" if HYPERPARAMETER_TUNE is False else "models/hyperparameter_tuning"
MODEL_PATH = f"{MODEL_DIR_NAME}/{MODEL_NAME}"
MODEL_WEIGHTS_PATH = f"{MODEL_PATH}/{MODEL_NAME}_weights.keras"
TEST_RESULT_DIR_NAME = f"{MODEL_PATH}/results"
# Training Parameter
SAVE_HISTORY = True
EPOCHS = 100
BATCH_SIZE = 16
TF_SEED = 42
LEARNING_RATE = 0.001
PATIENCE = 5
TRANSFER_LEARNING = True

# Data Parameter
IAM_DATASET_PATH = "dataset/iam_dataset"
WHAT_DATASET = "lines"
TRANSFER_DATASET_PATH = "dataset/transfer_dataset"