from flask import Flask, request, jsonify
import os
from pipeline import pipeline
import base64
from io import BytesIO
from PIL import Image

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
    try:
        image_decoded = base64.b64decode(image_data)
        with open('uploaded_image.jpg', "wb") as img:
            img.write(image_decoded)
    except Exception as e:
        return jsonify({'message': str(e)}), 500
    return pipeline()(directory, 'uploaded_image.jpg')

print("Pipeline loaded successfully.")
if __name__ == '__main__':
    app.run()
