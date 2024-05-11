"""
Author: Alexej Kravtschenko
Co-Author: Tim Harmling

Description: This file represents most of our tested models including the final model "build_model9v3". 
Every model uses a layer for Connectionist Temporal Classification (CTC) in Keras.
It is used to compute the CTC loss function and incorporate it into a Keras model.
"""
# Used imports
import keras
import tensorflow as tf

# A custom layer for Connectionist Temporal Classification (CTC) in Keras.
class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        """Invoked when the layer is called.

        Computes the CTC loss function.

        Args:
            y_true: The true labels.
            y_pred: The predicted labels.

        Returns:
            The predicted labels (y_pred).
        """
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        return y_pred


def build_model9v3_xl(img_width, img_height, char, lr_value):
    """Builds a handwriting recognition model.

    This function constructs a handwriting recognition model using a convolutional neural network
    followed by bidirectional LSTM layers and a CTC loss layer.

    Args:
        img_width (int): The width of the input image. Default: 1024
        img_height (int): The height of the input image. Default: 128
        char (int): The number of characters to classify.
        lr_value (float): The learning rate value.

    Returns:
        The constructed Keras model.
    """
    input_img = keras.Input(shape=(img_width, img_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))
    
    x = keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(input_img)
    x = keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv3")(x)
    x = keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv4")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = keras.layers.Dropout(0.5)(x) # from 0.2 to 0.5 
    
    new_shape = ((img_width // 4), (img_height // 4) * 128)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(128, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)
                                
    x = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True, dropout=0.25))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(x)

    x = keras.layers.Dense(char + 2, activation="softmax", name="dense2")(x)
    output = CTCLayer(name="ctc_loss")(labels, x)

    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="handwriting_recognizer")
    opt = keras.optimizers.Adam(lr_value)
    model.compile(optimizer=opt)
    
    return model

# GOAT
def build_model9v3(img_width, img_height, char, lr_value):
    """Builds a handwriting recognition model.

    This function constructs a handwriting recognition model using a convolutional neural network
    followed by bidirectional LSTM layers and a CTC loss layer.

    Args:
        img_width (int): The width of the input image. Default: 1024
        img_height (int): The height of the input image. Default: 128
        char (int): The number of characters to classify.
        lr_value (float): The learning rate value.

    Returns:
        The constructed Keras model.
    """
    input_img = keras.Input(shape=(img_width, img_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))
    
    x = keras.layers.Conv2D(48, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(input_img)
    x = keras.layers.Conv2D(96, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = keras.layers.Conv2D(48, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv3")(x)
    x = keras.layers.Conv2D(96, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv4")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = keras.layers.Dropout(0.5)(x) # from 0.2 to 0.5 
    
    new_shape = ((img_width // 4), (img_height // 4) * 96)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(128, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)
                                
    x = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True, dropout=0.25))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(x)

    x = keras.layers.Dense(char + 2, activation="softmax", name="dense2")(x)
    output = CTCLayer(name="ctc_loss")(labels, x)

    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="handwriting_recognizer")
    opt = keras.optimizers.Adam(lr_value)
    model.compile(optimizer=opt)
    
    return model

