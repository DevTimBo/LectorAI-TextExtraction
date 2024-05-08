import tensorflow as tf

class lr_scheduler:

    def __init__(self, initial_learning_rate, decay_steps, alpha, warmup_target, warmup_steps, name):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.warmup_target = warmup_target
        self.warmup_steps = warmup_steps
        self.name = name
        
    def get_scheduler(self):
        if self.name == "cosine_decay":
            return tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=self.initial_learning_rate,
                decay_steps=self.decay_steps,
                alpha=self.alpha,
                warmup_target=self.warmup_target,
                warmup_steps=self.warmup_steps,
                name="cosine_decay",
            )
        elif self.name == "cosine_decay_restarts":
            return tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=self.initial_learning_rate,
                first_decay_steps=self.decay_steps,
                alpha=self.alpha,
                warmup_target=self.warmup_target,
                warmup_steps=self.warmup_steps,
                name="cosine_decay_restarts",
            )
        elif self.name == "exponential_decay":
            return tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.initial_learning_rate,
                decay_steps=self.decay_steps,
                decay_rate=self.alpha,
                staircase=True,
                name="exponential_decay"
                )
        elif self.name == "inverse_time_decay":
            return tf.keras.optimizers.schedules.InverseTimeDecay(
                initial_learning_rate=self.initial_learning_rate,
                decay_steps=self.decay_steps,
                decay_rate=self.alpha,
                staircase=True,
                name="inverse_time_decay"
                )
        
    def __call__(self, step):
        