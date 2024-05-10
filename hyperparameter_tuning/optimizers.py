import tensorflow as tf

class optimizers:

    def __init__(self, learning_rate, name):
        self.learning_rate = learning_rate
        self.name = name
    def __call__(self):
        if self.name == "adam":
            return tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.name == "adamax":
             return tf.keras.optimizers.Adamax(learning_rate=self.learning_rate)
        elif self.name == "nadam":
             return tf.keras.optimizers.Nadam(learning_rate=self.learning_rate)
        elif self.name == "ftrl":
             return tf.keras.optimizers.Ftrl(learning_rate=self.learning_rate)
        elif self.name == "adagrad":
             return tf.keras.optimizers.Adagrad(learning_rate=self.learning_rate)