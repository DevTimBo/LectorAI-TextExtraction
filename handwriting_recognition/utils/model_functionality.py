import utils.load_transfer_data as load_transfer_data
import utils.load_data as load_data
import utils.tokenizer as tokenizer
from utils.config import *

import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import os
import re
import pickle

weights_keras_string = "_weights.keras"

def train_model(model, train_ds, val_ds, epochs, callbacks):
        """Trains the model and returns prediction model and training history.

        This function trains the provided model using the training and validation datasets.
        It also returns a prediction model and training history.

        Args:
            model: The Keras model to be trained.

        Returns:
            prediction_model: The model used for predictions.
            history: The training history.
        """
        prediction_model = tf.keras.models.Model(model.get_layer(name="image").input, model.get_layer(name="dense2").output)
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
        history.history["lr"] = model.optimizer.lr.numpy()
        return prediction_model, history

def load_model_and_weights(model_path: str, weight_path: str):
    print(model_path)
    if os.path.exists(model_path):
        return extract_model_and_weights(model_path, weight_path)
    print("No pre-trained model or weights found.")
    return None


def extract_model_and_weights(model_path: str, weight_path: str):
    print("Loading pre-trained model and weights...")
    model = tf.keras.models.load_model(model_path)
    model_weight_path = weight_path
    model.load_weights(model_weight_path)
    print("Model and weights loaded successfully.")
    return model


def save_model(model, model_name: str = MODEL_NAME, model_directory: str = MODEL_DIR_NAME):
    if not os.path.exists(model_directory):
        create_dir(model_directory)
    model_path = os.path.join(model_directory, "{model_name}".format(model_name=model_name))
    model.save(model_path)
    model.save_weights(
        os.path.join(model_path, f"{model_name}{weights_keras_string}"),
        overwrite=True,
        save_format=None,
        options=None,
    )
    json_string = model.to_json()

    with open(os.path.join(model_path, f"{model_name}.json"), "w") as f:
        f.write(json_string)

    data_to_save = (tokenizer.max_len, tokenizer.chars)
    with open(os.path.join(model_path, "handwriting_chars.pkl"), "wb") as file:
        pickle.dump(data_to_save, file)


def save_train_history(
    prediction_model,
    history,
    val_ds,
    model_name: str = MODEL_NAME,
    test_result_directory: str = TEST_RESULT_DIR_NAME,
    total_duration: int = 0,
    add_name: str = None,
):
    """Author: Alexej Kravtschenko (main) and Tim Harmling (wrote)

    Creates a new plot name based on existing names.
    This function generates a new plot name by appending a version number to the given model name.
    The version number is determined based on existing plot names in the directory.

    Args:
        model_name (str): The base model name.
        names (list): A list of existing plot names.
        format (str): The format string for the plot name.

    Returns:
        str: The new plot name.
    """
    if not os.path.exists(test_result_directory):
        create_dir(test_result_directory)

    files_with_model_name = [
        file for file in os.listdir(test_result_directory) if model_name in file
    ]
    metrics = history.history

    NAME = "{name}_{epoch}E_{height}H_{width}W_{loss}L_{val_loss}VL_{time}s".format(
        name=model_name,
        epoch=history.epoch[-1],
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        loss=round(metrics["loss"][-1], 2),
        val_loss=round(metrics["val_loss"][-1], 2),
        time=round(total_duration),
    )

    if add_name is not None:
        NAME = f"{NAME}_{add_name}"

    if files_with_model_name:
        new_name = create_new_plot_name(model_name, files_with_model_name, NAME)
        plot_history(history, new_name, test_result_directory, True)
        plot_evaluation(NAME, test_result_directory, True, val_ds, prediction_model)
    else:
        plot_history(history, NAME, test_result_directory, True)
        plot_evaluation(NAME, test_result_directory, True, val_ds, prediction_model)


def create_new_plot_name(model_name, names, format):
    pattern = r"\d+"
    max_number = 0
    for name in names:
        tmp_name = name.replace(model_name, "")
        number = int(re.findall(pattern, tmp_name)[0])
        if number > max_number:
            max_number = number

    new_model_name = f"{model_name}V_{str(max_number + 1)}"

    return format.replace(model_name, new_model_name)


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
                plt.title(f'Name: {name}')
                path = os.path.join(dir_path, f'{name}_history.png')
                plt.savefig(path)



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
                path = os.path.join(dir_path, f'{name}_result.png')
                plt.savefig(path)
        


def create_dir(path_to_dir: str):
    isExist = os.path.exists(path_to_dir)
    if not isExist:
        os.makedirs(path_to_dir)


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][
        0
    ][:, : load_data.max_len]
    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(tokenizer.num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text
