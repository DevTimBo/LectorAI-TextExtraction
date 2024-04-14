FROM python:3.9.19-alpine3.19

# Install system dependencies for opencv
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install system dependencies for pyzbar
RUN apt-get update && apt-get install -y \
    libzbar0 \
    && rm -rf /var/lib/apt/lists/*
    
# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r r.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
CMD ["gunicorn", "web_api:app", "-b", "0.0.0.0:80"]