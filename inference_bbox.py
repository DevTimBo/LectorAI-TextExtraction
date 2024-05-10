import cv2
import keras_cv
import numpy as np


CLASSES =[
    "parent_first_name",
    "parent_last_name",
    "parent_email",
    "parent_phone",

    "child_first_name",
    "child_last_name",
    "child_class",

    "address_street_and_number",
    "address_zip",
    "address_city",

    "ag_1",
    "ag_2",
    "ag_3",
]

NUM_CLASSES = len(CLASSES)
MODEL_WEIGHT_PATH = None
MODEL_IMAGE_WIDTH = 1024
MODEL_IMAGE_HEIGHT = 128

class bbox_model:
    def __init__(self):
        self.model = self.load_model_and_weights(MODEL_WEIGHT_PATH)   

    def load_model_and_weights(self, model_weight_path: str):
        """Loads a pre-trained model and its weights.

        This function, loads a pre-trained model and its weights
        from the specified directory. It checks if both the model and weights exist before loading.

        Returns:
            model: The pre-trained Keras model with loaded weights, if found.
        """
        backbone = keras_cv.models.YOLOV8Backbone.from_preset(
            "yolo_v8_xs_backbone_coco",
            load_weights=True 
        )
        model = keras_cv.models.YOLOV8Detector(
            num_classes=NUM_CLASSES, 
            bounding_box_format="xyxy",
            backbone=backbone,
            fpn_depth=1,
        )
        # No Weights yet
        # yolo.load_weights(model_weight_path)
        return model
    
    def inference(self, image_path):
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, (1536, 768))
        resized_image = np.expand_dims(resized_image, axis=0)
        predictions = self.model.predict(resized_image)
        boxes = predictions['boxes'][0]
        confidence = predictions['confidence'][0]
        classes = predictions['classes'][0]
        return boxes, confidence, classes

bbox_model = bbox_model()

if __name__ == "__main__":
    model = bbox_model()
    print(model.inference("data/example.png"))