import onnxruntime as ort
import tensorflow as tf
import json 
import numpy as np
import matplotlib.pyplot as plt
import cv2

from pathlib import Path
MODEL_PATH = "TOP_AMAZON_WORKER/mask_rcnn"
MODEL_WEIGHT_PATH = f"{MODEL_PATH}/Bilder-maskrcnn_resnet50_fpn_v2.onnx"
LOGGING = True

class bbox_model:
    def __init__(self):
        pass

    def get_color_map(self):
        checkpoint_dir_obj = Path(MODEL_PATH)
        colormap_path = list(checkpoint_dir_obj.glob('*colormap.json'))[0]
        with open(colormap_path, 'r') as file:
            colormap_json = json.load(file)    
        colormap_dict = {item['label']: item['color'] for item in colormap_json['items']}
        class_names = list(colormap_dict.keys())
        return class_names
    
    def inference(self, test_img, onnx_file_path, class_names, threshold=0.8, img_size=(1024, 1024)):
        original_height, original_width = test_img.shape[:2]
        input_img = tf.image.resize(test_img, img_size)
        input_img = tf.expand_dims(input_img, axis=0)
        input_img = tf.transpose(input_img, perm=[0, 3, 1, 2])
        input_img = np.array(input_img, np.float32) / 255.0
        
        session = ort.InferenceSession(onnx_file_path, providers=['GPUExecutionProvider', 'CPUExecutionProvider'])
        model_output = session.run(None, {"input": input_img})

        scores_mask = model_output[2] > threshold
        pred_bounding_boxes = (model_output[0][scores_mask])
        pred_labels = [class_names[int(idx)] for idx in model_output[1][scores_mask]]
        pred_scores = model_output[2]

        results = []
        for label, box, score in zip(pred_labels, pred_bounding_boxes, pred_scores):
            x_min, y_min, x_max, y_max = box
            x_min = int(x_min * original_width / img_size[1])
            y_min = int(y_min * original_height / img_size[0])
            x_max = int(x_max * original_width / img_size[1])
            y_max = int(y_max * original_height / img_size[0])
            box = [x_min, y_min, x_max, y_max]
            results.append((label, box, score.item()))
        
        if LOGGING is True:
            for label, box, score in results:
                print(f"Label: {label}, Box: {box}, Confidence: {score}")
                # maybe save? idk
        return results

    def __call__(self, image):
        class_names = self.get_color_map()
        return self.inference(image, MODEL_WEIGHT_PATH, class_names=class_names)

bbox_model = bbox_model()

if __name__ == "__main__":
    image = tf.io.read_file("data/Jason_Ad_2.jpg")
    image = tf.image.decode_png(image, channels=3)
    model = bbox_model(image)