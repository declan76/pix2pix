# Table of Contents
- [Table of Contents](#table-of-contents)
- [Data Cube](#data-cube)
  - [Overview](#overview)
    - [Why Preprocessing is Essential for pix2pix](#why-preprocessing-is-essential-for-pix2pix)
    - [Challenges of Dynamic Image Size Adjustment](#challenges-of-dynamic-image-size-adjustment)
  - [1. Single FITS File Processor](#1-single-fits-file-processor)
  - [2. Three Different FITS Files Processor](#2-three-different-fits-files-processor)
- [Pair Generation](#pair-generation)
      - [pix2pix Model Context](#pix2pix-model-context)
      - [Naming Convention](#naming-convention)
      - [How It Works](#how-it-works)
- [Data Set Splitter](#data-set-splitter)
- [Data Augmentation](#data-augmentation)

# Data Cube

## Overview
This repository contains Python scripts designed to preprocess FITS files for the pix2pix model. The primary goal is to transform the data into the specific format required by pix2pix, ensuring that the data is compatible and optimized for training.

There are two scripts available to preprocess the data:
1. single_fits_pre.py: Duplicate a single FITS file to create a three-channel image.  
2. three_fits_pre.py: Combine three different FITS files to create a three-channel image. 

**Important Note**: When running the code inside a Docker container, please ensure that you use relative paths for any file or directory references. Absolute paths that work on your local machine might not be recognized correctly within the Docker container environment. Using relative paths will ensure consistent behavior and prevent potential file not found errors.

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
**Script**: [single_fits_pre.py](https://github.com/declan76/pix2pix/blob/main/preprocessing/data_cube/single_fits_pre.py)

This script processes individual FITS files.
It normalizes the data based on the type of FITS file (magnetogram, intensity, or divergence).
The data is duplicated to fit into three channels.
The data is cropped to a size of 256x256 pixels.
The processed data is saved in a new directory, preserving the original filename.

**Run the Script**: Execute the script using the command:
```
/usr/bin/python3 /app/preprocessing/data_cube/single_fits_pre.py
```

## 2. Three Different FITS Files Processor
**Script**: [three_fits_pre.py](https://github.com/declan76/pix2pix/blob/main/preprocessing/data_cube/three_fits_pre.py)

This script processes three different FITS files to create a 3D data cube.
It normalizes the data based on the type of FITS file.
The script uses one file for each channel: magnetogram, intensity, and divergence.
The data is cropped to a size of 256x256 pixels.
The processed data cube is saved in a new directory with a filename indicating the active region and time interval.

**Run the Script**: Execute the script using the command:
```
/usr/bin/python3 /app/preprocessing/data_cube/three_fits_pre.py
```

# Pair Generation

**Script**: [pair_files.py](https://github.com/declan76/pix2pix/blob/main/preprocessing/pair_files.py)

This script is designed to process a directory of files with specific naming conventions related to active regions (AR) and time intervals (TI). It pairs files based on consecutive time intervals for the same active region and outputs the pairs to a CSV file. The primary motivation behind this pairing is to prepare input-target image pairs for the pix2pix image-to-image translation model.

**Run the Script**: Execute the script using the command:
```
/usr/bin/python3 /app/preprocessing/pair_files.py
```

#### pix2pix Model Context
The pix2pix model requires paired data for training, where each pair consists of an input image and a corresponding target (or real) image. This script aids in generating such pairs.

#### Naming Convention
Files should be named with an active region label (e.g., AR11158) and a time interval (e.g., TI+04, TI-4, or TI4). The script identifies and extracts these components from the filenames to determine the appropriate file pairs.

#### How It Works
1. **File Parsing**: The script scans the provided directory and uses regular expressions to extract the active region and time interval from each filename.
2. **Pairing Logic**: Files are grouped by their active region. Within each group, files are paired based on consecutive time intervals. For instance, if time intervals 1, 2, 3, and 4 are present for a specific active region, the pairs would be 1-2, 2-3, and 3-4. The script also ensures that the time intervals are in the correct order (e.g., 1-2, not 2-1). If a time interval is missing, the script will skip that pairing. For example, if the time intervals 1, 2, and 4 are present, the pairs would be 1-2, but the pairing 2-4 would be skipped because 3 is missing.
3. **CSV Output**: The identified file pairs are written to a CSV file. Each row in the CSV contains two filenames representing a pair: the first file is the input image, and the second file is the target/real image for the pix2pix model.

# Data Set Splitter
**Script**: [split_dataset.py](https://github.com/declan76/pix2pix/blob/main/preprocessing/split_data.py)

The Data Set Splitter is a utility designed to divide a dataset into training and testing subsets based on a specified ratio. This is crucial for machine learning models, as it allows for the evaluation of the model's performance on unseen data after training.

**Run the Script**: Execute the script using the command:
```
/usr/bin/python3 /app/preprocessing/split_data.py
```

# Data Augmentation
  - **Status**: Not developed.
  - **Details**:
    - **Flipping**: Images are mirrored along their vertical axis.
    - **Magnetic Field Adjustment**: Images are multiplied by -1 to maintain magnetic field polarity.
  - **Script**: [augmentation.py](https://github.com/declan76/pix2pix/blob/main/preprocessing/augmentation.py)