# Author: Jason Pranata
import keras
import tensorflow as tf

class optimizers:
     def __init__(self, learning_rate, name):
          self.learning_rate = learning_rate
          self.name = name
     def __call__(self):
          if self.name == "adam":
               return keras.optimizers.Adam(learning_rate=self.learning_rate)
          elif self.name == "adamax":
               return keras.optimizers.Adamax(learning_rate=self.learning_rate)
          elif self.name == "nadam":
               return keras.optimizers.Nadam(learning_rate=self.learning_rate)
          elif self.name == "adamw":
               return keras.optimizers.AdamW(learning_rate=self.learning_rate)
          elif self.name == "adagrad":
               return keras.optimizers.Adagrad(learning_rate=self.learning_rate)

class lr_scheduler:
     def __init__(self, initial_learning_rate: float = 0.001, decay_steps: int = 32*10, name: str = "cosine_decay"):
          self.initial_learning_rate = initial_learning_rate
          self.decay_steps = decay_steps
          self.name = name

     def __call__(self):
          if self.name == "cosine_decay":
               return tf.keras.optimizers.schedules.CosineDecay(
               initial_learning_rate=self.initial_learning_rate,
               decay_steps=self.decay_steps
          )
          elif self.name == "cosine_decay_restarts":
               return tf.keras.optimizers.schedules.CosineDecayRestarts(
               initial_learning_rate=self.initial_learning_rate,
               first_decay_steps=self.decay_steps
          )
          elif self.name == "exponential_decay":
               return tf.keras.optimizers.schedules.ExponentialDecay(
               initial_learning_rate=self.initial_learning_rate,
               decay_steps=self.decay_steps,
               decay_rate=0.1,
               staircase=True
               )
          elif self.name == "inverse_time_decay":
               return tf.keras.optimizers.schedules.InverseTimeDecay(
               initial_learning_rate=self.initial_learning_rate,
               decay_steps=self.decay_steps,
               decay_rate=0.1,
               staircase=True
               )
          elif self.name == "polynomial_decay":
               return tf.keras.optimizers.schedules.PolynomialDecay(
               initial_learning_rate=self.initial_learning_rate,
               decay_steps=self.decay_steps,
               end_learning_rate=0.00001,
               power=1.0,
               cycle=True
               )
