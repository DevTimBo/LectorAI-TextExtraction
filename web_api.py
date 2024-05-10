from flask import Flask, request, jsonify
import uuid
import os
from pipeline import pipeline

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
    for file in files:
        filename = f"{uuid.uuid4()}.png"
        file.save(os.path.join(directory, filename))
        return pipeline()(directory, filename)

if __name__ == '__main__':
    app.run()
    
