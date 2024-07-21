# Author Jason Pranata
import keras
import tensorflow as tf

#@tf.function
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