FROM ubuntu:latest

RUN apt update
RUN apt install python3 python3-pip -y

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install flask tensorflow==2.15 gunicorn

# Make port 80 available to the world outside this container
EXPOSE 80

# Run web_api.py when the container launches
CMD ["gunicorn", "web_api:app", "-b", "0.0.0.0:80"]