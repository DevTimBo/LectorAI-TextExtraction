FROM python:3.10.14-bullseye

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies for opencv
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install flask tensorflow==2.15 gunicorn opencv-python ultralytics tensorflow-probability==0.23.0 dill

# Make port 80 available to the world outside this container
EXPOSE 80

# Run web_api.py when the container launches
CMD ["gunicorn", "web_api:app", "-b", "0.0.0.0:80"]