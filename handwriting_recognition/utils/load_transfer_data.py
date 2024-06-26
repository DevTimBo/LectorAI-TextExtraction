"""
Author: Alexej Kravtschenko

Description: This file contains our method of loading the transfer data (post christmas).
Dependencies: Depending on the used notebook a different base_path needs to be used.
Every other element stays independent. 
"""
# Used imports
import os
from utils.config import TRANSFER_DATASET_PATH

characters = set() # Set to contain unique characters
max_len = 0 # Number of total characters used

cwd = os.getcwd()
last_part = os.path.basename(cwd)

# Path to the dataset for transferlearning
base_path = TRANSFER_DATASET_PATH
train_dataset_path = os.path.join(base_path, 'train')
val_dataset_path = os.path.join(base_path, 'val')

def read_data(path):
    """Reads image and label data from files.

    This function reads image and label data from files stored in a specified directory.
    It searches for JPEG image files and corresponding text files containing labels.
    It creates a list of tuples, where each tuple contains the file path to an image
    and its corresponding label.

    Returns:
        A list of tuples, where each tuple contains the file path to an image and its label.
    """
    data_list = []
    image_files = [f for f in os.listdir(path) if f.endswith('.jpg')]

    for image_file in image_files:
        image_name = os.path.splitext(image_file)[0]

        img_path = os.path.join(path, image_file)
        label_file = os.path.join(path, f"{image_name}.txt")

        if os.path.exists(label_file):
            try:
                with open(label_file, "r", encoding="utf-8") as file:
                    line = file.readline().strip()
            except UnicodeDecodeError as e:
                print(e)
                continue
            data_list.append((img_path, line))

    return data_list


def split_data(lines_list):
    """Splits the dataset into training, testing, and validation sets.

    This function splits a list of data samples into training, testing, and validation sets.
    It divides the data into 90% training, 5% validation, and 5% testing by default.

    Args:
        lines_list (list): A list of tuples containing image paths and labels.

    Returns:
        A tuple containing the training, testing, and validation sets.
    """
    split_idx = int(0.9 * len(lines_list))
    train_samples = lines_list[:split_idx]
    test_samples = lines_list[split_idx:]

    val_split_idx = int(0.5 * len(test_samples))
    validation_samples = test_samples
    test_samples = []
    return train_samples, test_samples, validation_samples


def get_image_paths_and_labels(samples):
    """Extracts image paths and corresponding labels from a list of samples.

    This function extracts image paths and corresponding labels from a list of samples
    and creates separate lists for image paths and labels.

    Args:
        samples (list): A list of tuples containing image paths and labels.

    Returns:
        Two lists: one containing image paths (X) and the other containing labels (y).
    """
    x_img_paths = []
    y_labels = []

    for img_path, label in samples:
        if os.path.exists(img_path):
            x_img_paths.append(img_path)
            y_labels.append(label)

    return x_img_paths, y_labels


def get_vocabulary_length(data):
    """Gets the length of the vocabulary and maximum sequence length from the dataset.

    This function calculates the length of the vocabulary (number of unique characters)
    and the maximum sequence length from the dataset.

    Args:
        data (list): A list of tuples containing image paths and labels.

    Returns:
        A tuple containing the vocabulary (list of unique characters) and the maximum sequence length.
    """
    global characters, max_len

    for _, label in data:
        for char in label:
            characters.add(char)

        max_len = max(max_len, len(label))

    characters = sorted(list(characters))

    print("Maximum length: ", max_len)
    print("Vocab size: ", len(characters))
    return characters, max_len


train_data = read_data(train_dataset_path)
val_data = read_data(val_dataset_path)
all_data = train_data + val_data
characters, max_len = get_vocabulary_length(all_data)

# Some utility functions
def print_samples():
    print(f"Total train samples: {len(train_data)}")
    print(f"Total validation samples: {len(val_data)}")
    
def get_train_data():
    
    train_path, train_label = get_image_paths_and_labels(train_data)
    
    return train_path, train_label

def get_validation_data():
    
    val_path, val_label = get_image_paths_and_labels(val_data)
    
    return val_path, val_label

    
