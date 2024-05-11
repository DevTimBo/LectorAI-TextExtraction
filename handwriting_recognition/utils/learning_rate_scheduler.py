import tensorflow as tf

class lr_scheduler:

    def __init__(self, initial_learning_rate: float = 0.001, decay_steps: int = 10000, name: str = "cosine_decay"):
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
                decay_rate=self.initial_learning_rate/100,
                staircase=True
                )
        elif self.name == "inverse_time_decay":
            return tf.keras.optimizers.schedules.InverseTimeDecay(
                initial_learning_rate=self.initial_learning_rate,
                decay_steps=self.decay_steps,
                decay_rate=self.initial_learning_rate/100
                )
        elif self.name == "polynomial_decay":
            return tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=self.initial_learning_rate,
                decay_steps=self.decay_steps
                )
