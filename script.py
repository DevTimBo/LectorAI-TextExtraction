import requests

url = 'http://localhost:80/inference'
file_path = "/mnt/c/Users/timBo/Desktop/Projects/LectorAI-TextExtraction/dataset/iam_dataset/lines/a01/a01-000u/a01-000u-00.png"

with open(file_path, 'rb') as f:
    files = {
        'files': (file_path, f, 'image/jpeg')  
    }
    response = requests.post(url, files=files)

if response.status_code == 200:
    print("Image uploaded successfully. Response:", response.json())
else:
    print(f"Failed to upload image. Status code: {response.status_code}, Response: {response.text}")
