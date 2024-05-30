from keras.models import load_model
import numpy as np
import os
import pickle
from keras.layers import StringLookup
import tensorflow as tf
import keras
import handwriting_recognition.utils.preprocess as preprocess

MODEL_PATH = "MAFIA_BOSS_LINES_NEW_DS_adam"
MODEL_WEIGHT_PATH = "MAFIA_BOSS_LINES_NEW_DS_adam/MAFIA_BOSS_LINES_NEW_DS_adam_weights.keras"
MODEL_CHARS_PATH = os.path.join(MODEL_PATH, "handwriting_chars.pkl")
MODEL_IMAGE_WIDTH = 1024
MODEL_IMAGE_HEIGHT = 128


class handwriting_model():
    def __init__(self):
        self.model = self.load_model_and_weights(MODEL_PATH, MODEL_WEIGHT_PATH)   

    def load_model_and_weights(self, model_path: str, model_weight_path: str):
        """Loads a pre-trained model and its weights.

        This function, loads a pre-trained model and its weights
        from the specified directory. It checks if both the model and weights exist before loading.

        Returns:
            model: The pre-trained Keras model with loaded weights, if found. Default: model9v3_xl
        """
        print(model_path)
        if os.path.exists(model_path):
            print("Loading pre-trained model and weights...")
            model = load_model(model_path)
            if os.path.exists(model_weight_path):
                model.load_weights(model_weight_path)
                print("Model and weights loaded successfully.")

            return model
        else:
            print("No pre-trained model or weights found.")
            return None
        
    def load_metadata(self, file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data[0], data[1]
    
    def decode_single_prediction(self, pred):
        max_len, char = self.load_metadata(MODEL_CHARS_PATH)

        # Mapping characters to integers.
        char_to_num = StringLookup(vocabulary=list(char), mask_token=None)

        # Mapping integers back to original characters.
        num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        result = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_len]
        result = tf.gather(result[0], tf.where(tf.math.not_equal(result[0], -1)))
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        return result

    def decode_batch_predictions(self,pred):
        max_len, char = self.load_metadata(MODEL_CHARS_PATH)
        # Mapping characters to integers.
        char_to_num = StringLookup(vocabulary=list(char), mask_token=None)

        # Mapping integers back to original characters.
        num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search.
        results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][
            0
        ][:, : max_len]
        # Iterate over the results and get back the text.
        output_text = []
        for res in results:
            res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
            res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text
        
    def inference(self, image):
        img_size=(MODEL_IMAGE_WIDTH, MODEL_IMAGE_HEIGHT)
        image = preprocess.distortion_free_resize(image, img_size)
        image = tf.cast(image, tf.float32) / 255.0

        prediction_model = keras.models.Model(self.model.get_layer(name="image").input, self.model.get_layer(name="dense2").output)
        preds = prediction_model.predict(tf.expand_dims(image, axis=0))
        pred_texts = self.decode_single_prediction(preds)

        return pred_texts.replace("|"," ")
    
    def inference_batch(self, images):
        img_size=(MODEL_IMAGE_WIDTH, MODEL_IMAGE_HEIGHT)
        preprocessed_images = []
        for image in images:
            image = preprocess.distortion_free_resize(image, img_size)
            image = tf.cast(image, tf.float32) / 255.0
            preprocessed_images.append(image)
        images = tf.stack(preprocessed_images, axis=0)
        prediction_model = keras.models.Model(self.model.get_layer(name="image").input, self.model.get_layer(name="dense2").output)
        preds = prediction_model.predict(images)
        pred_texts = self.decode_batch_predictions(preds)
        for pred in pred_texts:
            pred.replace("|"," ")

        return pred_texts

handwriting_model = handwriting_model()

if __name__ == "__main__":
    IMAGE_PATH = "dataset/transfer_dataset/train/0_5.jpg"
    image = tf.io.read_file(IMAGE_PATH)
    image = tf.image.decode_png(image, 1)
    IMAGE_PATH2 = "dataset/transfer_dataset/train/0_6.jpg"
    image2 = tf.io.read_file(IMAGE_PATH)
    image2 = tf.image.decode_png(image2, 1)
    pred = handwriting_model.inference_batch([image, image2])
    print(pred)