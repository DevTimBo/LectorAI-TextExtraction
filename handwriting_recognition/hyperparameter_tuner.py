# Author Jason Pranata

import utils.optimizers as opt
import utils.learning_rate_scheduler as lrs
import utils.tokenizer as tk
import utils.load_transfer_data as load_transfer_data
import utils.load_data as load_data
import utils.model_functionality as mf
import utils.models as models
from utils.config import *

import tensorflow as tf
import time
import json
import os

class hyperparameter_tuner:
    def __init__(self, initial_learning_rate: float = LEARNING_RATE):
        self.optimizers_list = ["adam", "adamw", "nadam", "adamax"]
        self.lrs_list = [""]#"cosine_decay", "cosine_decay_restarts", "exponential_decay", "inverse_time_decay", "polynomial_decay"]
        self.decay_steps_list = [""]
        self.learning_rate_list = [0.001]
        self.initial_learning_rate = initial_learning_rate

    def __call__(self, base_model = None):
        best_hyperparameters = {}
        x_train_img_paths, y_train_labels = load_transfer_data.get_train_data()
        x_val_img_paths, y_val_labels = load_transfer_data.get_validation_data()
        train_ds = tk.prepare_augmented_dataset(x_train_img_paths, y_train_labels, BATCH_SIZE)
        val_ds = tk.prepare_dataset(x_val_img_paths, y_val_labels, (IMAGE_WIDTH, IMAGE_HEIGHT), BATCH_SIZE)
        for optimizer_name in self.optimizers_list:
            for scheduler_name in self.lrs_list:
                for learning_rate in self.learning_rate_list:
                    # Model Name to be saved and for tensorboard/results plots
                    model_name = f"MAFIA_BOSS_LINES_NEW_DS_{optimizer_name}_measure_time"
                    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
                    tf.keras.callbacks.TensorBoard(log_dir=f"{MODEL_DIR_NAME}/logs/{model_name}")]
                    
                    # Load Model if u want to use different pre-trained models
                    model_path = f"{MODEL_DIR_NAME}/trained_with_lines_dataset/{MODEL_NAME}_{optimizer_name}_dataset_lines_0.001"
                    base_model = mf.load_model_and_weights(model_path, f"{model_path}/{MODEL_NAME}_{optimizer_name}_dataset_lines_0.001_weights.keras")
                    
                    #lr_scheduler = lrs.lr_scheduler(initial_learning_rate=learning_rate, decay_steps=decay_steps, name = scheduler_name)()
                    optimizer = opt.optimizers(learning_rate = learning_rate, name = optimizer_name)() #lr_scheduler, name = optimizer_name)()
                    base_model.compile(optimizer=optimizer)
                    start_time = time.time()
                    prediction_model, history = mf.train_model(base_model, train_ds, val_ds, EPOCHS, callbacks)
                    total_duration = time.time() - start_time
                    best_hyperparameters[(optimizer_name, scheduler_name)] = history.history["val_loss"][-1]
                    mf.save_model(
                        model=base_model,
                        model_name=model_name
                    )
                    mf.save_train_history(prediction_model, history, val_ds, model_name=model_name, 
                                        test_result_directory=f"{MODEL_DIR_NAME}/{model_name}/results", total_duration=total_duration)
        # Save and Rank results in JSON file
        sorted_hyperparameters = sorted(best_hyperparameters.items(), key=lambda item: item[1])
        json_data = json.dumps(sorted_hyperparameters, indent=4)
        file_path = f"{MODEL_DIR_NAME}/hyperparameters.json"
        os.makedirs(MODEL_DIR_NAME, exist_ok=True) 
        with open(file_path, "w") as json_file:
            json_file.write(json_data)

# Script
if __name__ == "__main__":
    #base_model = mf.load_model_and_weights(model_path="models/keras/Model9v3Words", weight_path="models/keras/Model9v3Words/Model9v3Words_weights.keras")
    #base_model = models.build_model9v3(IMAGE_WIDTH, IMAGE_HEIGHT, len(tk.char_to_num.get_vocabulary()), LEARNING_RATE)
    tuner = hyperparameter_tuner()
    tuner()
