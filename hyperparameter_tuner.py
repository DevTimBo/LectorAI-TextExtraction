import optimizers as opt
import learning_rate_scheduler as lrs
import tokenizer as tk
import load_transfer_data
import model_functionality as mf
from config import *

import tensorflow as tf
import time

class hyperparameter_tuner:
    def __init__(self, initial_learning_rate: float = LEARNING_RATE, decay_steps: int = 10000):
        self.optimizers_list = ["adam", "adamax", "nadam", "ftrl", "adagrad"]
        self.lrs_list = ["cosine_decay", "cosine_decay_restarts", "exponential_decay", "inverse_time_decay", "polynomial_decay"]
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps

    def __call__(self, model):
        x_train_img_paths, y_train_labels = load_transfer_data.get_train_data()
        x_val_img_paths, y_val_labels = load_transfer_data.get_validation_data()
        train_ds = tk.prepare_dataset(x_train_img_paths, y_train_labels, (IMAGE_WIDTH, IMAGE_HEIGHT), BATCH_SIZE)
        val_ds = tk.prepare_dataset(x_val_img_paths, y_val_labels, (IMAGE_WIDTH, IMAGE_HEIGHT), BATCH_SIZE)
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)]
        for optimizer_name in self.optimizers_list:
            for scheduler_name in self.lrs_list:
                lr_scheduler = lrs.lr_scheduler(initial_learning_rate=ilr, decay_steps=self.decay_steps, name = scheduler_name)()
                optimizer = opt.optimizers(learning_rate = lr_scheduler, name = optimizer_name)()
                tf.print(f"Optimizer: {optimizer}, Scheduler: {lr_scheduler}")
                model.compile(optimizer=optimizer)
                start_time = time.time()
                prediction_model, history = mf.train_model(model, train_ds, val_ds, EPOCHS, callbacks)
                total_duration = time.time() - start_time
                mf.save_model(
                    model=model,
                    model_name=f"{MODEL_NAME}_{optimizer_name}_{scheduler_name}"
                )
                mf.save_train_history(prediction_model, history, val_ds, model_name=MODEL_NAME, total_duration=total_duration, add_name=f"_{optimizer_name}_{scheduler_name}")

# Script
if __name__ == "__main__":
    model = mf.load_model_and_weights(model_path=MODEL_PATH, weight_path=MODEL_WEIGHTS_PATH)
    tuner = hyperparameter_tuner()
    tuner(model)
