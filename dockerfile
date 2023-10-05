# Use an official Tensorflow runtime as a parent image. 


# ***Comment out either the GPU or CPU version depending on your system.***
# FOR GPU VERSION: 
FROM tensorflow/tensorflow:2.13.0-gpu
# FOR CPU VERSION:
# FROM tensorflow/tensorflow:2.13.0

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies for pycairo
RUN apt-get update && apt-get install -y libcairo2-dev 

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 6006 available to the world outside this container
EXPOSE 6006