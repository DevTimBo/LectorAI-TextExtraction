FROM tensorflow/tensorflow:2.15.0-gpu

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies for opencv
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --ignore-installed --force flask gunicorn opencv-python numpy onnx onnxsim onnxruntime onnxruntime-gpu tensorflow-probability==0.23.0 pillow

# Make port 80 available to the world outside this container
EXPOSE 80

# Run web_api.py when the container launches
CMD ["gunicorn", "web_api:app", "-b", "0.0.0.0:80"]