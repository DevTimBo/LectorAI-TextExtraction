#Own Imports
import utils.load_data as load_data
import utils.load_transfer_data as load_transfer_data
import utils.models as models
import utils.model_functionality as model_functionality
#Imports
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import time
import os
import re
from keras.callbacks import History
from utils.config import *
import pickle
iam_history1, iam_history2, transfer_history1, transfer_history2, transfer_history3 = None, None, None, None, None 

import handwriting_recognition.utils.tokenizer as tokenizer

def main():
    # IAM Dataset
    x_train_img_paths, y_train_labels = load_data.train_data
    x_test_img_paths, y_test_labels = load_data.test_data
    x_val_img_paths, y_val_labels = load_data.val_data
    
    # Transfer Dataset
    x_train_transfer_img_paths, y_train_transfer_labels = load_transfer_data.train_data
    x_val_transfer_img_paths, y_val_transfer_labels = load_transfer_data.val_data

    #train_ds = tokenizer.prepare_dataset(x_train_img_paths, y_train_labels, (IMAGE_WIDTH,IMAGE_HEIGHT),BATCH_SIZE)
    val_ds = tokenizer.prepare_dataset(x_val_img_paths, y_val_labels,(IMAGE_WIDTH,IMAGE_HEIGHT),BATCH_SIZE)
    test_ds = tokenizer.prepare_dataset(x_test_img_paths, y_test_labels,(IMAGE_WIDTH,IMAGE_HEIGHT),BATCH_SIZE)
    aug_train_ds = tokenizer.prepare_augmented_dataset(x_train_img_paths, y_train_labels, BATCH_SIZE)
    char = len(tokenizer.char_to_num.get_vocabulary())
    
    # IAM Training
    ## Phase 1
    start_time = time.time()
    model = models.build_model9v3_xl(IMAGE_WIDTH, IMAGE_HEIGHT, char, LEARNING_RATE)
    prediction_model, iam_history1 = model_functionality.train_model(model, aug_train_ds, val_ds)

    # Delete old datasets of memory
    if TRANSFER_LEARNING:
        del aug_train_ds
        del val_ds
        del test_ds

        # Transfer Learning
        ## Prepare New Datasets
        aug_train_ds = tokenizer.prepare_augmented_dataset(x_train_transfer_img_paths, y_train_transfer_labels, BATCH_SIZE)
        val_ds = tokenizer.prepare_dataset(x_val_transfer_img_paths, y_val_transfer_labels,(IMAGE_WIDTH,IMAGE_HEIGHT),BATCH_SIZE)   
        ## Phase 1 Dense Training
        opt = keras.optimizers.Adam(LEARNING_RATE)
        model.compile(optimizer=opt)
        for layer in model.layers:
            if "dense2" in layer.name:
                layer.trainable = True
            else:
                layer.trainable = False
        prediction_model, transfer_history1 = model_functionality.train_model(model, aug_train_ds, val_ds)
        ## Phase 2 - Full Training
        for layer in model.layers:
            layer.trainable = True
        ## Phase 3 - Lower Learning Rate
        prediction_model, transfer_history2 = model_functionality.train_model(model, aug_train_ds, val_ds)
        opt = keras.optimizers.Adam(LEARNING_RATE/10)
        model.compile(optimizer=opt)
        prediction_model, transfer_history3 = model_functionality.train_model(model, aug_train_ds, val_ds)
        total_duration = time.time() - start_time

    # Combine Histories
    history = combine_n_histories(iam_history1, iam_history2, transfer_history1, transfer_history2, transfer_history3)

    # Saving Model, Chars, Results and History
    model_path = os.path.join(MODEL_DIR_NAME, "{model_name}".format(model_name=MODEL_NAME))
    TEST_RESULT_DIR_NAME = os.path.join(model_path, "results")
    if not os.path.exists(TEST_RESULT_DIR_NAME):
            model_functionality.create_dir(TEST_RESULT_DIR_NAME)
    metrics = history.history

    NAME = "{name}_{epoch}E_{height}H_{width}W_{loss}L_{val_loss}VL_{time}s".format(
        name=MODEL_NAME, epoch=history.epoch[-1], height=IMAGE_HEIGHT, width=IMAGE_WIDTH,
        loss=round(metrics['loss'][-1],2), val_loss=round(metrics['val_loss'][-1], 2), time=round(total_duration))

    if SAVE_HISTORY:
        model_functionality.plot_history(history, NAME, TEST_RESULT_DIR_NAME, True)
        model_functionality.plot_evaluation(NAME, TEST_RESULT_DIR_NAME, True, val_ds, prediction_model)

    if MODEL_SAVE:
        model_functionality.save_model(prediction_model, model_path, MODEL_NAME)


def combine_n_histories(*histories):
        new_history = History()
        new_history.history = {}
        new_history.epoch = []
        for history in histories:
            if history is None:
                continue
            new_history.epoch += history.epoch
            for key in history.history.keys():
                if key in new_history.history:
                    new_history.history[key] += history.history[key]
                else:
                    new_history.history[key] = history.history[key]
        return new_history


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    GPU_LIST = []
    for gpu in gpus:
        GPU_LIST.append(f"GPU:{gpu.name.split(':')[-1]}")
    strategy = tf.distribute.MirroredStrategy(GPU_LIST)
    with strategy.scope():
        main()
        print("Training Completed!")
