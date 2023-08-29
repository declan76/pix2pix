"""
Declan Keir 

29/08/2023
"""

import os
from astropy.io import fits
import numpy as np
from scipy.ndimage import zoom

def read_fits(file_path):
    """Read a FITS file and return its data."""
    with fits.open(file_path) as hdul:
        data = hdul[0].data
        if data is None:
            raise ValueError(f"Error: No data in FITS file: {file_path}")
    return data

def duplicate_data(data, channels=3):
    """Duplicate the FITS data to fit into the required channels."""
    duplicated_data = np.stack([data] * channels, axis=-1)
    return duplicated_data

def normalize_data(data, data_type):
    """Normalize the data based on the type of FITS file."""
    magnetogram_factor  = 4000.  # gauss
    intensity_factor    = 50000.   # ?
    divergence_factor   = 100.    # cm/s

    if data_type == 1:          # Magnetogram
        return data / magnetogram_factor
    elif data_type == 2:        # Intensity
        return data / intensity_factor
    elif data_type == 3:        # Divergence
        return data / divergence_factor
    else:
        raise ValueError(f"Unknown data type: {data_type}")

def resize_data(data, method=1, target_shape=(256, 256)):
    """Resize the data using either cropping or interpolation."""
    if method == 1:     # Cropping
        cropped_data = data[:target_shape[0], :target_shape[1]]
        return cropped_data
    elif method == 2:   # Interpolation
        y_zoom = target_shape[0] / data.shape[0]
        x_zoom = target_shape[1] / data.shape[1]
        return zoom(data, (y_zoom, x_zoom))
    else:
        raise ValueError(f"Unknown resize method: {method}")

def process_directory(input_dir, output_dir, data_type, resize_method):
    """Process all FITS files in the given directory."""
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".fits"):
            file_path = os.path.join(input_dir, file_name)
            
            # Read the FITS file
            data = read_fits(file_path)

            # Duplicate the data
            duplicated_data = duplicate_data(data)

            # Normalize the data
            normalized_data = normalize_data(duplicated_data, data_type)

            # Resize the data
            resized_data = resize_data(normalized_data, resize_method)

            # Save the processed data
            output_path = os.path.join(output_dir, file_name)
            hdu = fits.PrimaryHDU(data=resized_data)
            hdu.writeto(output_path, overwrite=True)

if __name__ == "__main__":
    input_dir = input("Enter the input directory path: ")
    output_dir = input("Enter the output directory path: ")
    data_type = int(input("Enter the type of FITS file (1 for magnetogram, 2 for intensity, 3 for divergence): "))
    resize_method = int(input("Choose the resize method (1 for cropping, 2 for interpolation): "))
    process_directory(input_dir, output_dir, data_type, resize_method)
