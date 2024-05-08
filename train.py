#Own Imports
import load_data
import load_transfer_data
import tokenizer
import models
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
from config import *
import pickle
iam_history1, iam_history2, transfer_history1, transfer_history2, transfer_history3 = None, None, None, None, None 

def main():
    load_data.print_samples(IAM_DATASET_PATH)
    x_train_img_paths, y_train_labels = load_data.get_train_data()
    x_test_img_paths, y_test_labels = load_data.get_test_data()
    x_val_img_paths, y_val_labels = load_data.get_validation_data()

    x_train_transfer_img_paths, y_train_transfer_labels = load_transfer_data.get_train_data()
    x_val_transfer_img_paths, y_val_transfer_labels = load_transfer_data.get_validation_data()
    
    #train_ds = tokenizer.prepare_dataset(x_train_img_paths, y_train_labels, (IMAGE_WIDTH,IMAGE_HEIGHT),BATCH_SIZE)
    val_ds = tokenizer.prepare_dataset(x_val_img_paths, y_val_labels,(IMAGE_WIDTH,IMAGE_HEIGHT),BATCH_SIZE)
    test_ds = tokenizer.prepare_dataset(x_test_img_paths, y_test_labels,(IMAGE_WIDTH,IMAGE_HEIGHT),BATCH_SIZE)
    aug_train_ds = tokenizer.prepare_augmented_dataset(x_train_img_paths, y_train_labels, BATCH_SIZE)
    char = len(tokenizer.char_to_num.get_vocabulary())

    # IAM Training
    ## Phase 1
    start_time = time.time()
    model = models.build_model9v4(IMAGE_WIDTH, IMAGE_HEIGHT, char, LEARNING_RATE)
    prediction_model, iam_history1 = train_model(model, aug_train_ds, val_ds)
    ## Phase 2 - Lower Learning Rate
    opt = keras.optimizers.Adam(LEARNING_RATE/10)
    model.compile(optimizer=opt)
    prediction_model, iam_history2 = train_model(model, aug_train_ds, val_ds)

    # Delete old datasets of memory
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
    prediction_model, transfer_history1 = train_model(model, aug_train_ds, val_ds)
    ## Phase 2 - Full Training
    for layer in model.layers:
        layer.trainable = True
    ## Phase 3 - Lower Learning Rate
    prediction_model, transfer_history2 = train_model(model, aug_train_ds, val_ds)
    opt = keras.optimizers.Adam(LEARNING_RATE/10)
    model.compile(optimizer=opt)
    prediction_model, transfer_history3 = train_model(model, aug_train_ds, val_ds)
    total_duration = time.time() - start_time
    history = combine_n_histories(iam_history1, iam_history2, transfer_history1, transfer_history2, transfer_history3)

    # Saving Model, Chars and History
    model_path = os.path.join(MODEL_DIR_NAME, "{model_name}".format(model_name=MODEL_NAME))
    TEST_RESULT_DIR_NAME = os.path.join(model_path, "results")
    if not os.path.exists(TEST_RESULT_DIR_NAME):
            create_dir(TEST_RESULT_DIR_NAME)
    metrics = history.history

    NAME = "{name}_{epoch}E_{height}H_{width}W_{loss}L_{val_loss}VL_{time}s".format(
        name=MODEL_NAME, epoch=history.epoch[-1], height=IMAGE_HEIGHT, width=IMAGE_WIDTH,
        loss=round(metrics['loss'][-1],2), val_loss=round(metrics['val_loss'][-1], 2), time=round(total_duration))

    if SAVE_HISTORY:
        plot_history(history, NAME, TEST_RESULT_DIR_NAME, True)
        plot_evaluation(NAME, TEST_RESULT_DIR_NAME, True, val_ds, prediction_model)

    if MODEL_SAVE:
        if not os.path.exists(MODEL_DIR_NAME):
            create_dir(MODEL_DIR_NAME)
        
        model.save(model_path)
        model.save_weights(os.path.join(model_path, f"{MODEL_NAME}_weights.keras"), overwrite=True, save_format=None, options=None)
        json_string = model.to_json()

        with open(os.path.join(model_path, f"{MODEL_NAME}.json"),'w') as f:
            f.write(json_string)

        data_to_save = (load_data.max_len, load_data.characters)
       
        with open(os.path.join(model_path, "handwriting_chars.pkl"), 'wb') as file:
            pickle.dump(data_to_save, file)



def train_model(model, train_ds, val_ds):
        prediction_model = keras.models.Model(model.get_layer(name="image").input, model.get_layer(name="dense2").output)
        early_stopping = EarlyStopping(patience=PATIENCE, restore_best_weights=True)
        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[early_stopping])    
        history.history["lr"] = model.optimizer.lr.numpy()
        return prediction_model, history

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

def combine_histories(history1, history2):
        new_history = History()
        new_history.history = {}
        new_history.epoch = history1.epoch + [e + max(history1.epoch) + 1 for e in history2.epoch]
        all_keys = set(history1.history.keys()).union(set(history2.history.keys()))
        for key in all_keys:
            new_history.history[key] = history1.history.get(key, []) + history2.history.get(key, [])
        return new_history

def plot_history(history, name, dir_path, save_fig):
    metrics = history.history
    _, ax1 = plt.subplots()
    ax1.plot(metrics['loss'], label='Training Loss', color='blue')
    ax1.plot(metrics['val_loss'], label='Validation Loss', color='red')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='black')
    ax1.tick_params('y', colors='black')
    ax1.legend(loc='upper left', bbox_to_anchor=(0.0, 0.95))  
    ax2 = ax1.twinx()
    ax2.plot(metrics['lr'], label='Learning Rate', color='green')
    ax2.set_ylabel('Learning Rate', color='black')
    ax2.set_yscale('log') 
    ax2.tick_params('y', colors='black')
    ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:1.0e}'))
    ax2.legend(loc='upper right', bbox_to_anchor=(1.0, 0.95))  
    if save_fig:
        plt.title('Name: '+name)
        path = os.path.join(dir_path, name + '_history.png')
        plt.savefig(path)

def create_dir(path_to_dir):
    isExist = os.path.exists(path_to_dir)
    if not isExist:
        os.makedirs(path_to_dir)

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :load_data.max_len]
    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(tokenizer.num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

def plot_evaluation(name, dir_path, save_fig, val_ds, prediction_model):
    for batch in val_ds.take(1):
        batch_images = batch["image"]
        _, ax = plt.subplots(4, 4, figsize=(32, 8))

        preds = prediction_model.predict(batch_images)
        pred_texts = decode_batch_predictions(preds)

        for i in range(min(16,BATCH_SIZE)):
            img = batch_images[i]
            img = tf.image.flip_left_right(img)
            img = tf.transpose(img, perm=[1, 0, 2])
            img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
            img = img[:, :, 0]

            title = f"Prediction: {pred_texts[i]}"
            ax[i // 4, i % 4].imshow(img, cmap="gray")
            ax[i // 4, i % 4].set_title(title)
            ax[i // 4, i % 4].axis("off")   
    if save_fig:
        path = os.path.join(dir_path, name + '_result.png')
        plt.savefig(path)


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    GPU_LIST = []
    for gpu in gpus:
        GPU_LIST.append(f"GPU:{gpu.name.split(':')[-1]}")
    strategy = tf.distribute.MirroredStrategy(GPU_LIST)
    with strategy.scope():
        main()
        print("Training Completed!")
