# Authors: Tim Harmling & Jason Pranata
from flask import jsonify
from inference_smartapp import handwriting_model
from inference_bbox import bbox_model
import cv2 
import numpy as np
import time

cropping_params = {
    "ad":                       {"left": 0, "bottom": 0},
    "ag":                       {"left": 0, "bottom": 0},
    "ad_erzieher_name":         {"left": 0.15, "bottom": 0},
    "ad_erzieher_vorname":      {"left": 0.15, "bottom": 0},
    "ad_erzieher_tel":          {"left": 0.2, "bottom": 0},
    "ad_erzieher_email":        {"left": 0.15, "bottom": 0},
    "schueler_name":            {"left": 0.15, "bottom": 0},
    "schueler_vorname":         {"left": 0.15, "bottom": 0},
    "schueler_klasse":          {"left": 0.125, "bottom": 0},
    "ad_neue_ad_str_haus_nr":   {"left": 0.275, "bottom": 0},
    "ad_neue_ad_plz":           {"left": 0.15, "bottom": 0},
    "ad_neue_ad_stadt":         {"left": 0.15, "bottom": 0},
    "ad_schueler_datum":        {"left": 0.2, "bottom": 0},
    "ag_auswahl_wahl_1":        {"left": 0.175, "bottom": 0},
    "ag_auswahl_wahl_2":        {"left": 0.175, "bottom": 0},
    "ag_auswahl_wahl_3":        {"left": 0.175, "bottom": 0},
    "ag_schueler_datum":        {"left": 0.275, "bottom": 0},
    "ad_schueler_unterschrift": {"left": 0.2, "bottom": 0.25}, 
    "ag_schueler_unterschrift": {"left": 0.2, "bottom": 0.25}, 
}

def crop(x1, y1, x2, y2, image, crop_left_percent, crop_bottom_percent):
    crop_left = int(crop_left_percent * (x2 - x1))
    crop_bottom = int(crop_bottom_percent * (y2 - y1))
    cropped_image = image[y1:y2 - crop_bottom, x1 + crop_left:x2]
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
    def _predict_handwriting_batch(self, images):
            return self.handwriting_model.inference_batch(images)
    
    def __call__(self, image, debug=False):
        start = time.time()
        results = self._predict_bounding_boxes(image)
        print("Bounding boxes predicted in", time.time() - start, "seconds.")

        full_img_np = image.numpy()
        full_img_np = cv2.cvtColor(full_img_np, cv2.COLOR_RGB2BGR)
        if debug:
            for label, box, score in results:
                x_min, y_min, x_max, y_max = box
                cv2.rectangle(full_img_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(full_img_np, f'{label}: {score:.2f}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imwrite("tempimages_api/document_with_bounding_boxes.png", full_img_np)
        print("Bounding boxes plotting done.")

        if len(results) == 0:
            print("Bounding Boxes not found.")
            # replace jsonify with None if testing with this script and not in docker image
            return jsonify({'message': "No Text Boxes Detected in the provided Image."}), 200


        cropped_images = []
        image_np = image.numpy()
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        for label, box, score in results:
            x_min, y_min, x_max, y_max = box
            crop_left_percent = 0
            crop_bottom_percent = 0
            if label not in cropping_params:
                continue
            params = cropping_params[label]
            crop_left_percent = float(params["left"])
            crop_bottom_percent = float(params["bottom"])
            cropped_image = crop(x_min, y_min, x_max, y_max, image_np, crop_left_percent, crop_bottom_percent)
            gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            gray_with_dim = np.expand_dims(gray, axis=2)
            cropped_images.append((label, gray_with_dim, score, box))
            if debug:
                cv2.imwrite(f"tempimages_api/{label}.png", gray)
        print("Cropped images successfully.")
        text_predictions = self._predict_handwriting_batch([image for label, image, score, box in cropped_images])

        predictions = []
        for i, prediction in enumerate(text_predictions):
            x1, y1, x2, y2 = cropped_images[i][3]
            predictions.append({"class": cropped_images[i][0], "prediction": prediction, "confidence": float(cropped_images[i][2]), "box": [x1, y1, x2, y2]})
        print("Returning JSON response.")
        # if testing with this script and not docker change to print instead
        #print(predictions)
        return jsonify({"predictions":predictions}) 

print("Pipeline loaded successfully.")

#pipeline = pipeline()
#image = tf.io.read_file(os.path.join("tempimages_api", "beispiel_form_covered.jpg"))
#image = tf.image.decode_png(image, channels=3)
#pipeline(image)

