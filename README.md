# Training Handwriting
## Environment Setup
Ensure you have Python 3.10.12 installed. Then, install the required dependencies using the following command:
```
pip install -r requirements
```
## Folder
All files related to training handwriting detection models are located in the /handwriting_recognition directory.
### Datasets
- Transfer Datasets: Located in /dataset/transfer_dataset as zip files. These need to be extracted and moved to the /train and /val folders for training purposes.
- IAM Dataset: Should be placed in the /dataset/iam_dataset directory.
## Creation of Transferdataset
If you have images and XML files in Pascal format, you can use the dataset_creator.ipynb notebook to create a transfer dataset for training your handwriting detection model.
# Build Docker Container
- Copy Model Weights (Mask RCNN and Handwriting Model) into Project
- Make sure the Model Path and Weight Path in inference_smartapp and inference_bbox are correct
## Build Command
Using Dockerfile
```
docker build -t ki_pipeline .
```
Using Docker Compose
```
docker compose build
```

# Load Docker Image if you have the .tar
```
docker load -i ki_pipeline.tar
```
# Start Docker Container
```
docker run -d -p 8080:80 --name ki_pipeline ki_pipeline
```

# Start Docker Container (with GPU)
```
docker run --gpus=all -d -p 8080:80 --name ki_pipeline ki_pipeline
```
# Check if GPU is working
## BBox Part
If the GPU is NOT WORKING
```
Specified provider 'CUDAExecutionProvider' is not in available provider names.
```
will be printed

## Handwriting Part
If the GPU is WORKING
```
device: 0, name: NVIDIA GeForce RTX 2080 SUPER, pci bus id: 0000:2b:00.0, compute capability: 7.5
```
will be printed (with your GPU Name)

# Debugging
Save debug images to host machine
```
docker cp container_id:/app/tempimages_api /path/on/host
```

# Endpoint
http://localhost:8080/inference 

POST JSON with Body:
```
{"image":"/9j/4QAWRXhpZgAATU0AKgAAAAgAAAAAAAD/4gHYSUNDX1BS}
```
Response looks like this:
```
{
    "predictions": [
        {
            "box": [
                386,
                1061,
                1170,
                1120
            ],
            "class": "ad_erzieher_email",
            "confidence": 0.9991801381111145,
            "prediction": "Iuedtke@gmx.de"
        },
        {
            "box": [
                322,
                1152,
                1220,
                1418
            ],
            "class": "schueler",
            "confidence": 0.9991145730018616,
            "prediction": "Lüdtke"
        }
}
```