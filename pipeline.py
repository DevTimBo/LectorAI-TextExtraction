from flask import jsonify
import tensorflow as tf
import os
from inference_smartapp import handwriting_model
from inference_bbox import bbox_model
import cv2 
import numpy as np

cropping_params = {
    "ad_erzieher_name":         {"left": 0.15, "bottom": 0},
    "ad_erzieher_vorname":      {"left": 0.15, "bottom": 0},
    "ad_erzieher_tel":          {"left": 0.2, "bottom": 0},
    "ad_erzieher_email":        {"left": 0.15, "bottom": 0},
    "schueler_name":            {"left": 0.15, "bottom": 0},
    "schueler_vorname":         {"left": 0.15, "bottom": 0},
    "schueler_klasse":          {"left": 0.125, "bottom": 0},
    "ad_neue_ad_str_haus_nr":   {"left": 0.275, "bottom": 0},
    "ad_neue_ad_plz":           {"left": 0.25, "bottom": 0},
    "ad_neue_ad_stadt":         {"left": 0.15, "bottom": 0},
    "ad_schueler_datum":        {"left": 0.2, "bottom": 0},
    "ag_auswahl_wahl_1":        {"left": 0.15, "bottom": 0},
    "ag_auswahl_wahl_2":        {"left": 0.15, "bottom": 0},
    "ag_auswahl_wahl_3":        {"left": 0.15, "bottom": 0},
    "ag_schueler_datum":        {"left": 0.3, "bottom": 0},
    "ad_schueler_unterschrift": {"left": 0.2, "bottom": 0}, 
    "ag_schueler_unterschrift": {"left": 0.2, "bottom": 0}, 
}

def crop(x1, y1, x2, y2, image, crop_left_percent, crop_bottom_percent):
    crop_left = int(crop_left_percent * (x2 - x1))
    crop_bottom = int(crop_bottom_percent * (y2 - y1))
    cropped_image = image[y1 + crop_bottom:y2, x1 + crop_left:x2]
    if cropped_image.size == 0:
        print("Cropped image is empty")
        return None
    return cropped_image

class pipeline:
    def __init__(self):
        self.bbox_model = bbox_model
        self.handwriting_model = handwriting_model

    def _predict_bounding_boxes(self, image):
        return self.bbox_model(image)

    def _predict_handwriting(self, image):
        return self.handwriting_model.inference(image)

    def __call__(self, directory, filename):
        image = tf.io.read_file(os.path.join(directory, filename))
        image = tf.image.decode_png(image, channels=3)
        print("Image loaded successfully.")
        results = self._predict_bounding_boxes(image)
        print("Bounding boxes predicted successfully.")
        classes, boxes, confidences = zip(*results)

        full_img_np = image.numpy()
        full_img_np = cv2.cvtColor(full_img_np, cv2.COLOR_RGB2BGR)
        for label, box, score in results:
            x_min, y_min, x_max, y_max = box
            cv2.rectangle(full_img_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(full_img_np, f'{label}: {score:.2f}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imwrite("tempimages_api/document_with_bounding_boxes.png", full_img_np)

        cropped_images = []
        image_np = image.numpy()
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        for box, sub_class in zip(*(boxes, classes)):
            x_min, y_min, x_max, y_max = box
            crop_left_percent = 0
            crop_bottom_percent = 0
            if sub_class not in cropping_params:
                continue
            params = cropping_params[sub_class]
            crop_left_percent = float(params["left"])
            crop_bottom_percent = float(params["bottom"])
            cropped_image = crop(x_min, y_min, x_max, y_max, image_np, crop_left_percent, crop_bottom_percent)
            cropped_images.append(cropped_image)
        print("Cropped images successfully.")
        
        text_predictions = []
        for i, img in enumerate(cropped_images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_with_dim = np.expand_dims(gray, axis=2)
            prediction = handwriting_model.inference(gray_with_dim)
            text_predictions.append(prediction)
            cv2.imwrite(f"tempimages_api/cropped_{i}.png", img)
            print("Text predicted successfully")
        
        predictions = []
        for i, prediction in enumerate(text_predictions):
            x1, y1, x2, y2 = boxes[i]
            predictions.append({"class": classes[i], "prediction": prediction, "confidence": float(confidences[i]), "box": [x1, y1, x2, y2]})
        print("Returning JSON response.")
        # if testing with this script change to print instead
        # print(predictions)
        return jsonify({"predictions":predictions}) 

print("Pipeline loaded successfully.")

#pipeline = pipeline()
#pipeline("tempimages_api", "beispiel_form.jpg")
