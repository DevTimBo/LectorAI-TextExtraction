import optimizers as opt
import learning_rate_scheduler as lrs
import tokenizer as tk
import load_transfer_data
import model_functionality as mf
from config import *

import tensorflow as tf
import time

class hyperparameter_tuner:
    def __init__(self, model, initial_learning_rate, decay_steps):
        self.optimizers_list = ["adam", "adamax", "nadam", "ftrl", "adagrad"]
        self.lrs_list = ["cosine_decay", "cosine_decay_restarts", "exponential_decay", "inverse_time_decay", "polynomial_decay"]
        self.model = model
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps

    def __call__(self):
        x_train_img_paths, y_train_labels = load_transfer_data.get_train_data()
        x_val_img_paths, y_val_labels = load_transfer_data.get_validation_data()
        train_ds = tk.prepare_dataset(x_train_img_paths, y_train_labels, (IMAGE_WIDTH, IMAGE_HEIGHT), BATCH_SIZE)
        val_ds = tk.prepare_dataset(x_val_img_paths, y_val_labels, (IMAGE_WIDTH, IMAGE_HEIGHT), BATCH_SIZE)
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)]
        for optimizer in self.optimizers_list:
            for scheduler in self.lrs_list:
                current_optimizer = opt.optimizers(lrs.lr_scheduler(self.initial_learning_rate, self.decay_steps, scheduler), optimizer)
                model = self.model.compile(optimizer=current_optimizer)
                start_time = time.time()
                prediction_model, history = mf.train_model(model, train_ds, val_ds, EPOCHS, callbacks)
                total_duration = time.time() - start_time
                mf.save_model(
                    model=model,
                    model_name=f"{MODEL_NAME}_{optimizer}_{scheduler}"
                )
                mf.save_train_history(history, model_name=MODEL_NAME,total_duration=total_duration,
                                    add_name=f"_{optimizer}_{scheduler}", prediction_model=prediction_model)
                

# Script
if __name__ == "__main__":
    model = mf.load_model_and_weights()
    tuner = hyperparameter_tuner(model, INITIAL_LEARNING_RATE, DECAY_STEPS)
    tuner()
