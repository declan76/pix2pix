# Data Cube

## Overview

This repository contains Python scripts designed to preprocess FITS files for the pix2pix model. The primary goal is to transform the data into the specific format required by pix2pix, ensuring that the data is compatible and optimized for training.

There are two scripts available to preprocess the data:
1. `single_fits_processor.py`: Duplicate a single FITS file to create a three-channel image.
2. `three_fits_processor.py`: Combine three different FITS files to create a three-channel image.

### Why Preprocessing is Essential for pix2pix

The pix2pix model requires input data to be in a specific format:

- **Three Channels (RGB)**: The pix2pix model is inherently designed to process images that follow the standard RGB format, which consists of three channels: Red, Green, and Blue. Even if the original data sources, such as FITS files, don't naturally come in this three-channel format, it's essential to adapt them accordingly. This adaptation ensures that the data aligns with the model's expectations, enabling it to function optimally and deliver accurate results.

- **Size (256x256 pixels)**: The standard input size for the pix2pix model is 256x256 pixels. This size strikes a balance between resolution and computational efficiency.

### Challenges of Dynamic Image Size Adjustment

While it's theoretically possible to increase the image size to 512x512 pixels or even larger, doing so comes with significant challenges:

- **Memory Consumption**: Larger image sizes exponentially increase the memory requirements. This can make the model inaccessible for users without high-end GPUs.

- **Training Time**: Training on larger images would take considerably longer, potentially making the process impractical for many applications.

- **Hyperparameter Refactoring**: The model's hyperparameters, optimized for 256x256 images, would need extensive adjustments. This is not a trivial task and requires a deep understanding of the model's architecture and behavior.

- **Overfitting**: The model might become more prone to overfitting, especially if the dataset isn't large enough.

## 1. Single FITS File Processor

#### Filename: `single_fits_processor.py`

This script processes individual FITS files.
It normalizes the data based on the type of FITS file (magnetogram, intensity, or divergence).
The data is duplicated to fit into three channels.
The script provides options to resize the image either by cropping or by interpolation.
The processed data is saved in a new directory, preserving the original filename.

## 2. Three Different FITS Files Processor

#### Filename: `three_fits_processor.py`

This script processes three different FITS files to create a 3D data cube.
It normalizes the data based on the type of FITS file.
The script uses one file for each channel: magnetogram, intensity, and divergence.
The data is cropped to a size of 256x256 pixels.
The processed data cube is saved in a new directory with a filename indicating the active region and time interval.
Additional functions are provided to calculate the distance to the disk center and to remove a 2D plane from the data.
