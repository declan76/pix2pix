# Pix2Pix - FITS

This code base provides a streamlined system for the Pix2Pix model, optimized for image-to-image translation with [FITS](https://fits.gsfc.nasa.gov/fits_home.html) files. Leveraging the TensorFlow framework, the implementation is dockerized for consistent performance across different platforms. Comprehensive documentation guides users through each step, from pre-processing to post-processing, simplifying the process of training and evaluating the model.


- This code is based on the [TensorFlow](https://www.tensorflow.org/tutorials/generative/pix2pix) implementation of Pix2Pix. 
- The original Pix2Pix paper can be found at [Image-to-image translation with conditional adversarial networks](https://arxiv.org/abs/1611.07004).

> Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1125-1134).

--- 



## Table of Contents

- [Pix2Pix - FITS](#pix2pix---fits)
  - [Table of Contents](#table-of-contents)
  - [System Overview](#system-overview)
    - [I. Pre-processing Phase](#i-pre-processing-phase)
    - [II. Main Application Phase](#ii-main-application-phase)
    - [III. Post-processing Phase](#iii-post-processing-phase)
  - [Class Diagrams](#class-diagrams)
    - [Preprocessing](#preprocessing)
    - [Main Application](#main-application)
    - [Post-processing](#post-processing)
  - [Visual Studio Code](#visual-studio-code)
    - [Essential Extensions for VS Code](#essential-extensions-for-vs-code)
    - [Container Attachment](#container-attachment)
  - [Docker](#docker)
    - [An Overview of Docker Containerization](#an-overview-of-docker-containerization)
      - [Isolation Principle](#isolation-principle)
      - [File Integration and Access](#file-integration-and-access)
    - [Installation](#installation)
      - [Ubuntu and Debian-based Linux Distros](#ubuntu-and-debian-based-linux-distros)
          - [1. Uninstall Old Docker Versions](#1-uninstall-old-docker-versions)
          - [2. Install Docker Using the APT Repository](#2-install-docker-using-the-apt-repository)
          - [3. Install Docker Engine](#3-install-docker-engine)
          - [4. Addressing Permission Issues](#4-addressing-permission-issues)
      - [Windows](#windows)
      - [macOS](#macos)
    - [Setup](#setup)
      - [CPU-Only Environments on Linux, Windows, and macOS](#cpu-only-environments-on-linux-windows-and-macos)
        - [1. Create a Docker Image from the Dockerfile](#1-create-a-docker-image-from-the-dockerfile)
        - [2. Build Docker Container from the Image](#2-build-docker-container-from-the-image)
      - [GPU-Accelerated Environments on Ubuntu and Debian-based Linux Distros](#gpu-accelerated-environments-on-ubuntu-and-debian-based-linux-distros)
        - [Prerequisites](#prerequisites)
          - [1. NVIDIA Container Toolkit Installation](#1-nvidia-container-toolkit-installation)
          - [1.1. Install NVIDIA Container Toolkit](#11-install-nvidia-container-toolkit)
          - [1.2. Generate a CDI Specification](#12-generate-a-cdi-specification)
          - [1.3. Setting up NVIDIA Container Toolkit](#13-setting-up-nvidia-container-toolkit)
          - [2. Pull TensorFlow GPU Docker Image](#2-pull-tensorflow-gpu-docker-image)
          - [3. Pull CUDA Docker Image](#3-pull-cuda-docker-image)
        - [5. Build Docker Container](#5-build-docker-container)
          - [5.1. Create a Docker Image from the Dockerfile](#51-create-a-docker-image-from-the-dockerfile)
          - [5.2. Build Docker Container from the Image](#52-build-docker-container-from-the-image)
        - [GPU Monitoring](#gpu-monitoring)
    - [Docker Management](#docker-management)
  - [Training](#training)
    - [I. Running the Training Script](#i-running-the-training-script)
    - [II. Model Configuration](#ii-model-configuration)
    - [III. Training Progress](#iii-training-progress)
    - [IV. Monitoring with TensorBoard](#iv-monitoring-with-tensorboard)
  - [Evaluation](#evaluation)
    - [Running the Evaluation Script](#running-the-evaluation-script)



## System Overview
### I. Pre-processing Phase
This phase prepares the data for the pix2pix model. Detailed steps and scripts related to pre-processing can be found [here](https://github.com/declan76/pix2pix/tree/main/preprocessing#readme).

**1. Data Augmentation**:
  - **Status**: Not developed.
  - **Details**:
    - **Flipping**: Images are mirrored along their vertical axis.
    - **Magnetic Field Adjustment**: Images are multiplied by -1 to maintain magnetic field polarity.
  - **Script**: [augmentation.py](https://github.com/declan76/pix2pix/blob/main/preprocessing/augmentation.py)

**2. Data Cube Creation**:
  - **Purpose**: Generate a three-channel fits file. Two approaches are available:
      - **Option 1**: Use a single fits file. 
        **Script**: [single_fits_pre.py](https://github.com/declan76/pix2pix/blob/main/preprocessing/data_cube/single_fits_pre.py).
      - **Option 2**: Use three separate fits files. 
       **Script**: [three_fits_pre.py](https://github.com/declan76/pix2pix/blob/main/preprocessing/data_cube/three_fits_pre.py).
        
        **Note**: A corresponding post-processing script has not been developed.

**3. Input and Target Fits Pairing**:
  - **Purpose**: Generate a CSV file with fits file pairs using the Active Region ID (AR) and time step from the fits filename. 
  - **Script**: [pair_files.py](https://github.com/declan76/pix2pix/blob/main/preprocessing/pair_files.py).

**4. Dataset Splitting**:
  - **Purpose**: Divide the dataset into training and testing subsets. Creates separate training and testing directories with associated pairs.csv files. The split ratio is user-defined.
  - **Script**: [split_dataset.py](https://github.com/declan76/pix2pix/blob/main/preprocessing/split_data.py)

### II. Main Application Phase
**5. Model Training**:
  - **Purpose**: Train the model using a version of pix2pix adapted for fits files.
  - **Features**:
    - **Checkpoints**: Save model states and sample images at intervals defined in config/hyperparameters.yaml.
    - **Logging**: Training data, logs, and hyperparameters are saved in a timestamped directory in the experiments folder.
  - **Storage**: Results are saved in the timestamped experiments directory.
  - **Script**: [main.py](https://github.com/declan76/pix2pix/blob/main/src/main.py) 

**6. Model Evaluation**:
  - **Purpose**: Test the trained model using Mean Squared Error (MSE).
  - **Outputs**:
    - A CSV file with MSE values.
    - A comparison collage: input images at t1, target images at t2, predicted images at t2, and error images.
    - MSE visualizations: Box plots and histograms.
    -  A PDF report summarizing the evaluation results.
  - **Storage**: Results are saved in the evaluation sub-folder in the timestamped experiments directory.
  - **Script**: [main.py](https://github.com/declan76/pix2pix/blob/main/src/main.py) 

### III. Post-processing Phase
**7. Data Conversion**:
   - **Purpose**: Convert the three-channel fits file back to individual fits files.
        - **Option 1**: Convert the predicted data cube to a single fits file by averaging channels and denormalizing based on input type. **Script**: [single_fits_post.py](https://github.com/declan76/pix2pix/blob/main/postprocessing/single_fits_post.py).
        - **Option 2**: This feature, which will be the counterpart to preprocessing/data_cube/three_fits_pre.py script. 
        **Note**: This script is yet to be developed.

![Process Flow](diagrams/process-flow.png)

---

## Class Diagrams
### Preprocessing 
![Preprocessing](diagrams/pre.png)


### Main Application 
![Main Application](diagrams/main.png)


### Post-processing 
![Post-processing](diagrams/post.png)

---

## Visual Studio Code

 While not mandatory, it is a versatile Integrated Development Environment (IDE) that offers a Docker extension, simplifying many Docker-related tasks.

Download Visual Studio Code [here](https://code.visualstudio.com/download).

### Essential Extensions for VS Code
1. [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) - Containers: Enables containers to open in a dedicated VS Code window.
2. [Docker](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker) - Enhances VS Code's integration with Docker.
3. [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) - Provides support for Python development.

### Container Attachment
Once you've set up Docker using this [guide](https://github.com/declan76/pix2pix#docker).

Attach the Docker container to VS Code for an integrated development experience. Right-click on the container and choose "Attach Visual Studio Code." This opens a new VS Code window linked to the container. 
**Note**: The Python extension must be reinstall inside this window due to the container's isolated environment.

**Opening the Application**: After attaching the container, select "Open Folder" in VS Code and navigate to the /app directory. This is the root directory for the application. 

## Docker 

### An Overview of Docker Containerization
Docker offers a platform for containerization, allowing developers to encapsulate an application and its dependencies into a singular unit, termed a 'container'. This ensures uniformity in application behavior across diverse environments, from local development machines to production servers.

#### Isolation Principle
Containers operate in an isolated environment on the host system. This encapsulation ensures that processes within a container remain segregated from the host system and other containers. It's analogous to compartmentalizing applications within distinct, secure silos on a single machine.

#### File Integration and Access
To utilize resources from the host system within a Docker container, one must explicitly transfer them into the container's filesystem. Once integrated, these resources are accessed using container-relative paths, such as /app/path/to/file, ensuring consistent referencing irrespective of the container's deployment environment.

--- 

### Installation
#### Ubuntu and Debian-based Linux Distros
For a detailed installation guide, refer to the official Docker documentation [here](https://docs.docker.com/engine/install/ubuntu/). 

###### 1. Uninstall Old Docker Versions
Before you can install Docker Engine, you must first make sure that any conflicting packages are uninstalled.

- Run the following command to uninstall all conflicting packages:
    ```
    for pkg in docker.io docker-doc docker-compose podman-docker containerd runc; do sudo apt-get remove $pkg; done
    ```
###### 2. Install Docker Using the APT Repository
Before you install Docker Engine for the first time on a new host machine, you need to set up the Docker repository. Afterward, you can install and update Docker from the repository.

- Update the apt package index and install packages to allow apt to use a repository over HTTPS:
    ```
    sudo apt-get update
    sudo apt-get install ca-certificates curl gnupg
    ```
- Add Docker’s official GPG key:
    ```
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
    ```
- Use the following command to set up the repository:
    ```
    echo \
    "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
    "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    ```
**Note**: If you use an Ubuntu derivative distro, such as Linux Mint, you may need to use UBUNTU_CODENAME instead of VERSION_CODENAME.

- Update the apt package index:
    ```
    sudo apt-get update
    ```
###### 3. Install Docker Engine
- Now, install Docker Engine along with some additional packages:
    ```
    sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    ```
- To verify the installation, run a test image:
    ```
    sudo docker run hello-world
    ```
    This command downloads a test image and runs it in a container. If you see a confirmation message, Docker Engine has been successfully installed.

###### 4. Addressing Permission Issues
If you encounter errors when trying to run Docker without sudo, it's due to the Docker user group not having any users. To resolve this:
- Add your user to the Docker group:
  ```
  sudo usermod -aG docker $USER
  ```
- Either log out and log back in to apply the changes or use the following command for immediate application:
  ```
  newgrp docker
  ```
- Verify that you can run docker commands without sudo:
  ```
  docker run hello-world
  ```
---
  
#### Windows
For a detailed installation guide, refer to the official Docker documentation [here](https://docs.docker.com/desktop/install/windows-install/).

---

#### macOS
For a detailed installation guide, refer to the official Docker documentation [here](https://docs.docker.com/desktop/install/mac-install/).

---

### Setup
#### CPU-Only Environments on Linux, Windows, and macOS
##### 1. Create a Docker Image from the Dockerfile
**Note**: Uncomment "FROM tensorflow/tensorflow:2.13.0" in the Dockerfile, and comment out "FROM tensorflow/tensorflow:2.13.0-gpu".
- Navigate to the directory containing the Dockerfile:
    ```
    cd /path/to/dockerfile
    ```
- Build the Docker image:
    ```
    docker build -t pix2pix-cpu-image .
    ```
    This command builds a Docker image named pix2pix-cpu-image from the Dockerfile in the current directory.
##### 2. Build Docker Container from the Image
To build a Docker container from the image:
- **Linux**:
    ```
    docker run -it --net=host --name pix2pix-cpu-container -v $(pwd):/app pix2pix-cpu-image bash
    ```
- **Windows and macOS**:
    ```
    docker run -it --name pix2pix-cpu-container -v $(pwd):/app pix2pix-cpu-image bash
    ```
This command creates a Docker container named pix2pix-cpu-container from the pix2pix-cpu-image image and runs it in interactive mode. It also mounts the current directory to the /app directory in the container.

---

#### GPU-Accelerated Environments on Ubuntu and Debian-based Linux Distros
**Important**: Ensure your system is equipped with an NVIDIA GPU that supports CUDA. If not, you'll need to follow the CPU version of this guide.

##### Prerequisites
- **NVIDIA Drivers**: Before you get started, make sure you have installed the NVIDIA driver for your Linux distribution. The recommended way to install drivers is to use the package manager for your distribution but other installer mechanisms are also available (e.g. by downloading .run installers from NVIDIA Driver Downloads).

###### 1. NVIDIA Container Toolkit Installation
For a detailed walkthrough, refer to the official NVIDIA guide [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#nvidia-drivers).

###### 1.1. Install NVIDIA Container Toolkit
- To generate Container Device Interface (CDI) specifications for NVIDIA devices on your system, you need the base components of the NVIDIA Container Toolkit.
    ```
    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit-base
    ```
- To confirm the installation of the NVIDIA Container Toolkit CLI (nvidia-ctk), run:
    ```
    nvidia-ctk --version
    ```
###### 1.2. Generate a CDI Specification
- To generate a CDI specification that references all devices:
    ```
    sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
    ```
- To view the names of the generated devices:
  ```
  grep "  name:" /etc/cdi/nvidia.yaml
  ```
###### 1.3. Setting up NVIDIA Container Toolkit
- Set up the package repository and GPG key:
    ```
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    ```
- Update the package listing and install the nvidia-container-toolkit:
    ```
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    ```
- Configure the Docker daemon to recognize the NVIDIA Container Runtime:
    ```
    sudo nvidia-ctk runtime configure --runtime=docker
    ```
- Restart the Docker daemon:
    ```
    sudo systemctl restart docker
    ```
- Test the setup by running a base CUDA container:
    ```
    sudo docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
    ```

###### 2. Pull TensorFlow GPU Docker Image
- To pull the 2.13.0 TensorFlow GPU image:
    ```
    docker pull tensorflow/tensorflow:2.13.0-gpu
    ```

###### 3. Pull CUDA Docker Image
- To pull the 12.2.0 CUDA image:
    ```
    docker pull nvidia/cuda:12.2.0-devel-ubuntu20.04
    ```
- To verify the GPU:
    ```
    lspci | grep -i nvidia
    ```
- To confirm the installation of nvidia-docker and CUDA:
    ```
    sudo docker run --gpus all --rm nvidia/cuda:12.2.0-devel-ubuntu20.04 nvidia-smi
    ```

##### 5. Build Docker Container
**Note**: Uncomment "FROM tensorflow/tensorflow:2.13.0-gpu" in the Dockerfile, and comment out "FROM tensorflow/tensorflow:2.13.0".
###### 5.1. Create a Docker Image from the Dockerfile
- Navigate to the directory containing the Dockerfile:
    ```
    cd /path/to/dockerfile
    ```
- Build the Docker image:
    ```
    docker build -t pix2pix-gpu-image .
    ```
    This command builds a Docker image named pix2pix-gpu-image from the Dockerfile in the current directory.
###### 5.2. Build Docker Container from the Image
- To build a Docker container from the image:
    ```
    docker run -it --net=host --gpus=all --name pix2pix-gpu-container -v $(pwd):/app pix2pix-gpu-image bash
    ```
    This command creates a Docker container named pix2pix-gpu-container from the pix2pix-gpu-image image and runs it in interactive mode with the host network and all GPUs enabled. It also mounts the current directory to the /app directory in the container.

##### GPU Monitoring
Monitoring your GPU is important for many machine learning applications:
- **Thermal Management**: ML tasks can be computationally intensive and lead to increased GPU temperatures. Monitoring helps in preventing overheating, which can disrupt long training sessions. Different GPU models, like the A100 and the 2080, have varying optimal operating temperatures. It's essential to be aware of the recommended temperature range for your specific GPU model. To determine the safe operating temperature for your GPU:
    1. Google the official specifications of your GPU model. For instance, search for "NVIDIA A100 safe operating temperature" or "RTX 2080 recommended temperature range."
    2. Refer to the manufacturer's documentation or official website for precise details.
    3. As a general rule of thumb, it's always better to operate at temperatures below the maximum threshold. 
- **Resource Management**: Deep learning models, in particular, can consume significant GPU memory. Keeping an eye on memory usage ensures that you're not exceeding available resources, which can lead to training failures or reduced model performance.

To monitor GPU metrics such as temperature, utilization, and memory usage:
``` 
nvidia-smi
```
To monitor GPU metrics in real-time you can download tool such as nvtop
  
To install nvtop:
```
sudo apt install nvtop
```
To run nvtop:
```
nvtop
``` 

---

### Docker Management
You can manage Docker containers and images either through your IDE or the terminal:
- List currently running Docker containers:
    ```
    docker ps
    ```
- List all Docker containers, including stopped ones:
    ```
    docker ps -a
    ```
- To stop a running container:
    ```
    docker stop [CONTAINER_NAME/ID]
    ```
- To remove a container:
    ```
    docker rm [CONTAINER_NAME/ID]
    ```
- List all Docker images:
    ```
    docker images
    ```
- Remove a Docker image:
    ```
    docker rmi [IMAGE_NAME/ID]
    ```

--- 

## Training 
Training the Pix2Pix model is a straightforward process. Follow the steps below to ensure a smooth training experience.

### I. Running the Training Script
To initiate the training process, run the [main.py](https://github.com/declan76/pix2pix/blob/main/src/main.py) script:
```
/usr/bin/python3 /app/src/main.py
```
When prompted, select 't' for training mode.

### II. Model Configuration
The model's hyperparameters and configurations are stored in a YAML file, which allows for easy adjustments without altering the main codebase.

Location: The configuration file can be found at [hyperparameters.yaml](https://github.com/declan76/pix2pix/blob/main/config/hyperparameters.yaml).

**Key Parameters**:
- **BUFFER_SIZE**: This parameter determines the number of images loaded into memory at once. It's essential for the Pix2Pix model as it affects the shuffling of the dataset. A larger buffer size ensures better shuffling at the cost of increased memory usage. Defualt value is 400.
- **BATCH_SIZE**: This parameter specifies the number of training examples utilized in one iteration. A batch size of 1 means that the model is trained using one example at a time. Default value is 1.
- **STEPS**: The training process will halt once this number of steps is reached. It essentially defines the total number of training iterations. Default value is 200,000.
- **SAVE_FREQ**: This parameter determines the frequency (in terms of steps) at which the model's state is saved as a checkpoint and a sample image is generated. For instance, a value of 1000 means a checkpoint is saved every 1000 steps. Default value is 5000.

### III. Training Progress
- **Terminal Output**: During training, the terminal provides detailed information about the model's progress. Every 1000 steps, a comprehensive update is printed, including loss values and other relevant metrics. Additionally, a dot is printed every 10 steps as a visual indicator of ongoing progress.
- **Storage**: All relevant training data, including logs, checkpoints, generated images, the dataset used, and the current configuration file, are stored in a timestamped directory: experiment/{datetime}.
- **Safe Termination**: If you need to interrupt the training process, use ctrl + c. This ensures that the current model state is saved as a checkpoint before the program exits.

### IV. Monitoring with TensorBoard

TensorBoard provides real-time visualization and monitoring of the training process, offering insights into various metrics and model performance.

- **Activation**:
    Open a new terminal. Run the following command:
    ```
    tensorboard --logdir=path/to/logs/fit
    ```
    Open a web browser and navigate to http://localhost:6006.

- **Further Reading**: For a more in-depth guide on using TensorBoard to visualize your model's performance, refer to [this documentation](https://pytext.readthedocs.io/en/master/visualize_your_model.html).

---

## Evaluation
The evaluation process is similar to training. Follow the steps below to ensure a smooth evaluation experience.

### Running the Evaluation Script
To initiate the evaluation process, run the [main.py](https://github.com/declan76/pix2pix/blob/main/src/main.py) script:
```
/usr/bin/python3 /app/src/main.py
```
When prompted, select 'e' for evaluation mode.