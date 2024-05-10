from flask import jsonify
import os
import tensorflow as tf
from inference_smartapp import handwriting_model
from inference_bbox import bbox_model, CLASSES

class pipeline:
    def __init__(self):
        self.bbox_model = bbox_model()
        self.handwriting_model = handwriting_model()
        
    def _predict_bounding_boxes(self, image):
        return self.bbox_model.inference(image)
    
    def _predict_handwriting(self, image):
        return self.handwriting_model.inference(image)
    
    def __call__(self, directory, filename):
        boxes, confidences, classes = self._predict_bounding_boxes(os.path.join(directory, filename))

        image = tf.io.read_file(os.path.join(directory, filename))
        image = tf.image.decode_png(image, 1)
        predictions = []

        for box, cl, conf in zip(boxes, classes, confidences):
            print(conf)
            if conf < 0.529:
                continue
            try:
                cropped_image = tf.image.crop_to_bounding_box(image, int(box[1]), int(box[0]), int(box[3]-box[1]), int(box[2]-box[0]))
                prediction = self._predict_handwriting(cropped_image)
                predictions.append({
                    "class": CLASSES[cl],
                    "prediction": prediction
                })
            except Exception as e:
                print("Bad Box!")
        return jsonify({"predictions":predictions})
