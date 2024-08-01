# Author: Jason Pranata
import keras
import tensorflow as tf

def optimizers(learning_rate, name):
     if name == "adam":
          return keras.optimizers.Adam(learning_rate=learning_rate)
     elif name == "adamax":
          return keras.optimizers.Adamax(learning_rate=learning_rate)
     elif name == "nadam":
          return keras.optimizers.Nadam(learning_rate=learning_rate)
     elif name == "adamw":
          return keras.optimizers.AdamW(learning_rate=learning_rate)
     elif name == "adagrad":
          return keras.optimizers.Adagrad(learning_rate=learning_rate)

def lr_schedulers(initial_learning_rate: float = 0.001, decay_steps: int = 32*10, name: str = "cosine_decay"):
     if name == "cosine_decay":
          return tf.keras.optimizers.schedules.CosineDecay(
          initial_learning_rate=initial_learning_rate,
          decay_steps=decay_steps
     )
     elif name == "cosine_decay_restarts":
          return tf.keras.optimizers.schedules.CosineDecayRestarts(
          initial_learning_rate=initial_learning_rate,
          first_decay_steps=decay_steps
     )
     elif name == "exponential_decay":
          return tf.keras.optimizers.schedules.ExponentialDecay(
          initial_learning_rate=initial_learning_rate,
          decay_steps=decay_steps,
          decay_rate=0.1,
          staircase=True
          )
     elif name == "inverse_time_decay":
          return tf.keras.optimizers.schedules.InverseTimeDecay(
          initial_learning_rate=initial_learning_rate,
          decay_steps=decay_steps,
          decay_rate=0.1,
          staircase=True
          )
     elif name == "polynomial_decay":
          return tf.keras.optimizers.schedules.PolynomialDecay(
          initial_learning_rate=initial_learning_rate,
          decay_steps=decay_steps,
          end_learning_rate=0.00001,
          power=1.0,
          cycle=True
          )
