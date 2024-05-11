import utils.optimizers as opt
import utils.learning_rate_scheduler as lrs
import utils.tokenizer as tk
import utils.load_transfer_data as load_transfer_data
import utils.model_functionality as mf
from utils.config import *

import tensorflow as tf
import time
import json

class hyperparameter_tuner:
    def __init__(self, initial_learning_rate: float = LEARNING_RATE, decay_steps: int = 10000):
        self.optimizers_list = ["adam", "adamax", "nadam", "ftrl", "adagrad"]
        self.lrs_list = ["cosine_decay", "cosine_decay_restarts", "exponential_decay", "inverse_time_decay", "polynomial_decay"]
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps

    def __call__(self, model):
        best_hyperparameters = {}
        x_train_img_paths, y_train_labels = load_transfer_data.get_train_data()
        x_val_img_paths, y_val_labels = load_transfer_data.get_validation_data()
        train_ds = tk.prepare_augmented_dataset(x_train_img_paths, y_train_labels, BATCH_SIZE)
        val_ds = tk.prepare_dataset(x_val_img_paths, y_val_labels, (IMAGE_WIDTH, IMAGE_HEIGHT), BATCH_SIZE)
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)]
        for optimizer_name in self.optimizers_list:
            for scheduler_name in self.lrs_list:
                lr_scheduler = lrs.lr_scheduler(initial_learning_rate=self.initial_learning_rate, decay_steps=self.decay_steps, name = scheduler_name)()
                optimizer = opt.optimizers(learning_rate = lr_scheduler, name = optimizer_name)()
                model.compile(optimizer=optimizer)
                start_time = time.time()
                prediction_model, history = mf.train_model(model, train_ds, val_ds, EPOCHS, callbacks)
                total_duration = time.time() - start_time
                best_hyperparameters[(optimizer_name, scheduler_name)] = history.history["val_loss"][-1]
                model_name = f"{MODEL_NAME}_{optimizer_name}_{scheduler_name}"
                mf.save_model(
                    model=model,
                    model_name=model_name
                )
                mf.save_train_history(prediction_model, history, val_ds, model_name=model_name, 
                                    test_result_directory=f"{MODEL_DIR_NAME}/{model_name}/results", total_duration=total_duration)
        sorted_hyperparameters = sorted(best_hyperparameters.items(), key=lambda item: item[1])
        json_data = json.dumps(sorted_hyperparameters, indent=4)
        file_path = f"{TEST_RESULT_DIR_NAME}/hyperparameters.json"
        with open(file_path, "w") as json_file:
            json_file.write(json_data)

# Script
if __name__ == "__main__":
    model = mf.load_model_and_weights(model_path=MODEL_PATH, weight_path=MODEL_WEIGHTS_PATH)
    tuner = hyperparameter_tuner()
    tuner(model)
