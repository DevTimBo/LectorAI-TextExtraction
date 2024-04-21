import matplotlib.pyplot as plt
import cv2
import os
from collections import namedtuple
from keras.models import load_model

MODEL_PATH = None
MODEL_WEIGHT_PATH = None
MODEL_CHARS_PATH = None
MODEL_IMAGE_WIDTH = 1024
MODEL_IMAGE_HEIGHT = 128

class bbox_model:
    def __init__(self):
        self.model = self.load_model_and_weights(MODEL_PATH, MODEL_WEIGHT_PATH)   

    def load_model_and_weights(self, model_path: str, model_weight_path: str):
        """Loads a pre-trained model and its weights.

        This function, loads a pre-trained model and its weights
        from the specified directory. It checks if both the model and weights exist before loading.

        Returns:
            model: The pre-trained Keras model with loaded weights, if found.
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
    
    def crop(xmin, ymin, xmax, ymax, image_path):
        image = cv2.imread(image_path)
        xmin = int(round(xmin))
        ymin = int(round(ymin))
        xmax = int(round(xmax))
        ymax = int(round(ymax))
        imgCropped = image[ymin:ymax, xmin:xmax]
        return imgCropped
    
    # only to show the cropped images
    def plot_cropped_images(self, image_path, boxes, sub_classes):
        ImageInfo = namedtuple('ImageInfo', ['image', 'sub_class','value'])
        images_info_cropped = []
        for i,box in enumerate(boxes):
            xmin, ymin, xmax, ymax = box
            imgCropped = self.crop(xmin, ymin, xmax, ymax, image_path)
            image_info = ImageInfo(image=imgCropped,sub_class=sub_classes[i],value="")
            images_info_cropped.append(image_info)
            plt.axis("off")
            plt.imshow(imgCropped)
            plt.show()
    
    def inference(self, image):
        # Predict the bounding boxes
        boxes = self.model.predict(image)
        cropped_images = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            imgCropped = self.crop(xmin, ymin, xmax, ymax, image)
            cropped_images.append(imgCropped)
        
        return cropped_images