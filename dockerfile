# Use an official Python runtime as a parent image
# FROM python:3.11.4

FROM tensorflow/tensorflow:2.13.0-gpu

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 6006 available to the world outside this container
EXPOSE 6006

# Run main.py when the container launches
# CMD ["python", "src/main.py"]

