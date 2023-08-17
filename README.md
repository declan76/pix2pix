# Pix2Pix 
- This code is based on the [TensorFlow](https://www.tensorflow.org/tutorials/generative/pix2pix) implementation of Pix2Pix. 
- The original Pix2Pix paper can be found at [Image-to-image translation with conditional adversarial networks](https://arxiv.org/abs/1611.07004).

  > Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1125-1134).

## Docker 
**Visual Studio Code**:  While not mandatory, it is a versatile Integrated Development Environment (IDE) that offers a Docker extension, simplifying many Docker-related tasks.
- Download Visual Studio Code [here](https://code.visualstudio.com/download).
- Access its Docker extension [here](https://code.visualstudio.com/docs/containers/overview).

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
- To monitor GPU metrics such as temperature, utilization, and memory usage:
    ```
    nvidia-smi
    ```
- To monitor GPU metrics in real-time you can download tool such as nvtop
  
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
