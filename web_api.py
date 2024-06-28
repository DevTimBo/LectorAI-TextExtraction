from flask import Flask, request, jsonify
import os
from pipeline import pipeline
import base64
from io import BytesIO
from PIL import Image
import tensorflow as tf

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
    data = request.json
    if data is None or 'image' not in data:
        return jsonify({'message': 'No image provided'}), 400
    image_data = data['image']
    if 'debug' not in data:
        debug = False
    else:
        debug = bool(data['debug'])
        print("Debug mode activated.", debug)
    try:
        FILENAME = 'uploaded_image.jpg'
        image_decoded = base64.b64decode(image_data)
        with open(os.path.join(directory, FILENAME), "wb") as img:
            img.write(image_decoded)

        image = tf.io.read_file(os.path.join(directory, FILENAME))
        image = tf.image.decode_png(image, channels=3)
        # Testing with rotated metadata images
        #image = tf.image.rot90(image, k=3)
        print("Image loaded successfully.")
    except Exception as e:
        return jsonify({'message': str(e)}), 500
    try:
        response = pipeline()(image, debug)
        return response
    except Exception as e:
        return jsonify({'message': str(e)}), 500
    

print("Pipeline loaded successfully.")
if __name__ == '__main__':
    app.run()
