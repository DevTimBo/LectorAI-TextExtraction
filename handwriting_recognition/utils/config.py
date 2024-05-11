# Model Parameter
MODEL_SAVE = True
MODEL_NAME = "mafia_merger_acquisition"
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 128
HYPERPARAMETER_TUNE = True

# Directory Parameter
MODEL_DIR_NAME = "models/keras"
MODEL_PATH = f"{MODEL_DIR_NAME}/{MODEL_NAME}"
MODEL_WEIGHTS_PATH = f"{MODEL_PATH}/{MODEL_NAME}_weights.keras"
# Training Parameter
SAVE_HISTORY = True
EPOCHS = 1
BATCH_SIZE = 4
TF_SEED = 42
LEARNING_RATE = 0.001
PATIENCE = 5
TRANSFER_LEARNING = False

# Data Parameter
IAM_DATASET_PATH = "dataset/iam_dataset"
WHAT_DATASET = "lines"
TRANSFER_DATASET_PATH = "dataset/transfer_dataset"