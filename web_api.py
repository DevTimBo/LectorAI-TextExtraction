from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import uuid
import os
import tensorflow as tf
from inferenz_smartapp import handwriting_model
from inferenz_bbox import bbox_model, CLASSES

app = Flask(__name__)

directory = os.path.abspath("tempimages_api")

def create_directory():
    """
    Create the directory if it doesn't exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def clear_directory():
    """
    Clear the contents of the directory.
    """
    for file in os.listdir(directory):
        os.remove(os.path.join(directory, file))
        
create_directory()
clear_directory()

@app.route('/inference', methods=['POST'])
def process_image():
    if 'files' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    files = request.files.getlist('files')
    predictions = []
    for file in files:
        filename = f"{uuid.uuid4()}.png"
        file.save(os.path.join(directory, filename))

        boxes, confidences, classes = bbox_model.inference(os.path.join(directory, filename))

        image = tf.io.read_file(os.path.join(directory, filename))
        image = tf.image.decode_png(image, 1)
        predictions = []

        for box, cl, conf in zip(boxes, classes, confidences):
            print(conf)
            if conf < 0.529:
                continue
            try:
                cropped_image = tf.image.crop_to_bounding_box(image, int(box[1]), int(box[0]), int(box[3]-box[1]), int(box[2]-box[0]))
                prediction = handwriting_model.inference(cropped_image)
                predictions.append({
                    "class": CLASSES[cl],
                    "prediction": prediction
                })
            except Exception as e:
                print("Bad Box!")
    return jsonify({"predictions":predictions})

if __name__ == '__main__':
    app.run()
    
