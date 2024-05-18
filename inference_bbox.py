import cv2
import numpy as np
from ultralytics import YOLO
import cv2


MODEL_WEIGHT_PATH = "try/sub/best.pt"
MODEL_IMAGE_WIDTH = 1024
MODEL_IMAGE_HEIGHT = 128

class bbox_model:
    def __init__(self):
        self.model = self.load_model_and_weights(MODEL_WEIGHT_PATH)   

    def load_model_and_weights(self, model_weight_path: str):
        model = YOLO(model_weight_path)  
        return model
    
    def inference(self, image):
        results = self.model(image)
        result = results[0].boxes
        classes = result.cls
        confidences = result.conf
        boxes = result.xywh
        mapped_classes = [self.model.names[int(cls)] for cls in classes]
        return boxes, confidences, mapped_classes
       
bbox_model = bbox_model()

if __name__ == "__main__":
    model = bbox_model()
    print(model.inference("data/example.png"))