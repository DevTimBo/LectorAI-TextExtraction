from flask import jsonify
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
    #print(cropped_image.shape)
    if cropped_image.size == 0:
        print("Cropped image is empty")
        return None
    return cropped_image

class pipeline:
    def __init__(self):
        self.bbox_model = bbox_model
        self.handwriting_model = handwriting_model

    def _predict_bounding_boxes(self, image):
        return self.bbox_model.inference(image)

    def _predict_handwriting(self, image):
        return self.handwriting_model.inference(image)

    def __call__(self, directory, filename):
        image = cv2.imread(os.path.join(directory, filename))
        boxes, confidences, classes = self._predict_bounding_boxes(image)

        cropped_images = []
        for box, map_class in zip(boxes, classes):
            box = box.cpu()
            x, y, w, h = np.array(box)
            x1, y1, x2, y2 = int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)

            crop_left_percent = 0
            crop_bottom_percent = 0
            if map_class not in cropping_params:
                continue

            #print(map_class, cropping_params[map_class])
            params = cropping_params[map_class]
            crop_left_percent = float(params["left"])
            crop_bottom_percent = float(params["bottom"])
            imgCropped = crop(x1, y1, x2, y2, image, crop_left_percent, crop_bottom_percent)
            cropped_images.append(imgCropped)

        text_predictions = []
        for i, img in enumerate(cropped_images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_with_dim = np.expand_dims(gray, axis=2)
            prediction = handwriting_model.inference(gray_with_dim)
            text_predictions.append(prediction)
            cv2.imwrite(f"tempimages_api/cropped_{i}.png", img)

        predictions = []
        for i, prediction in enumerate(text_predictions):
            box = boxes[i].cpu()
            x, y, w, h = np.array(box)
            x1, y1, x2, y2 = int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)
            predictions.append({"class": classes[i], "prediction": prediction, "confidence": float(confidences[i].cpu()), "box": [x1, y1, x2, y2]})
        return jsonify({"predictions":predictions})
