# Authors: Tim Harmling
# File contains the tokenize function to convert the labels into a format that the model can understand
#
# Can be used for both the IAM dataset and the transfer learning dataset
# but the right load_data or load_transfer_data file has to be imported

from keras.layers import StringLookup
import tensorflow as tf
import utils.preprocess as preprocess
import numpy as np

AUTOTUNE = tf.data.AUTOTUNE
from utils.config import *

img_size = (IMAGE_WIDTH, IMAGE_HEIGHT)

# Load Data
import utils.load_transfer_data as load_transfer_data 
import utils.load_data as load_data
max_len = max(load_transfer_data.max_len, load_data.max_len)

chars = set(list(load_transfer_data.characters) + list(load_data.characters))
chars = sorted(list(chars))

char_to_num = StringLookup(vocabulary=list(chars), mask_token=None)
num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

padding_token = 99

def vectorize_label(label):
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    length = tf.shape(label)[0]
    pad_amount = max_len - length
    label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=padding_token)
    return label


def process_images_labels(image_path, label):
    image = preprocess.preprocess_image(image_path, img_size)
    label = vectorize_label(label)
    return {"image": image, "label": label}


def prepare_dataset(image_paths, labels, img_size_config, batch_size_new):
    global img_size
    img_size = img_size_config
    global batch_size
    batch_size = batch_size_new
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
        process_images_labels, num_parallel_calls=AUTOTUNE
    )
    return dataset.batch(batch_size).cache().prefetch(AUTOTUNE)


def prepare_data(image_paths, labels):
    processed_data = [process_images_labels(image_path, label) for image_path, label in zip(image_paths, labels)]

    # Separate the processed data into x_train and y_train
    x_train = np.array([item['image'] for item in processed_data])
    y_train = np.array([item['label'] for item in processed_data])

    return x_train, y_train


def prepare_augmented_dataset(image_paths, labels, batch_size_new):
    # Prepare Dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
        lambda x, y: (process_images_labels(x, y)["image"], process_images_labels(x, y)["label"]),
        num_parallel_calls=AUTOTUNE
    )

    data_augmentation1 = tf.keras.Sequential([
        tf.keras.layers.RandomContrast(0.25, seed=42),
        tf.keras.layers.RandomBrightness(0.25, value_range=(0, 1), seed=42)
    ])
    data_augmentation2 = tf.keras.Sequential([
        tf.keras.layers.RandomContrast(0.25, seed=42),
        tf.keras.layers.RandomBrightness(0.25, value_range=(0, 1), seed=42),
        tf.keras.layers.RandomZoom((0.1, 0.2), seed=42),
    ])
    data_augmentation3= tf.keras.Sequential([
        tf.keras.layers.RandomContrast(0.25, seed=42),
        tf.keras.layers.RandomBrightness(0.25, value_range=(0, 1), seed=42),
        tf.keras.layers.RandomZoom((0, 0.1), seed=42, fill_mode='constant', fill_value=255),
        tf.keras.layers.RandomTranslation(height_factor=(0, 0.01), width_factor=(0, 0.01), seed=42, fill_mode='constant', fill_value=255),
        tf.keras.layers.RandomRotation(0.005, seed=42)
    ])
    # Apply Augmentation
    dataset = dataset.map(lambda x, y: (data_augmentation1(x, training=True), y), num_parallel_calls=AUTOTUNE)

    # Separate image and label
    dataset = dataset.map(lambda x, y: {"image": x, "label": y}, num_parallel_calls=AUTOTUNE)

    return dataset.batch(batch_size_new).cache().prefetch(AUTOTUNE)
