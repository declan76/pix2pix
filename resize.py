import os
from astropy.io import fits
import numpy as np
from scipy.ndimage import zoom

def resize_fits_image(input_path, output_path, new_shape=(256, 256)):
    # Read the FITS file
    with fits.open(input_path) as hdul:
        data = hdul[1].data  # Accessing the second HDU
        header = hdul[1].header

        # Ensure we're accessing the data correctly
        print(f"Original shape of {input_path}: {data.shape}")
        
        # Calculate the zoom factors for each dimension
        y_zoom = new_shape[0] / data.shape[0]
        x_zoom = new_shape[1] / data.shape[1]
        
        # Resize the image data using interpolation
        resized_data = zoom(data, (y_zoom, x_zoom))
        
        # Update the header with the new shape
        header['NAXIS1'] = new_shape[1]
        header['NAXIS2'] = new_shape[0]
        
        # Save the resized image to the output path
        fits.writeto(output_path, resized_data, header)


def print_file_shape(directory):
    """
    Prints the name and shape of each FITS file in the given directory.
    
    Parameters:
    - directory (str): Path to the directory.
    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and filename.endswith('.fits'):
            with fits.open(file_path) as hdul:
                data = hdul[0].data
                data_shape = data.shape
                print(f"File: {filename}, Shape: {data_shape}")


def main():
    source_dir = 'data'  # Current directory
    target_dir = os.path.join(source_dir, '256')
    
    # Create the target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Loop through all files in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith('.fits'):
            input_path = os.path.join(source_dir, filename)
            output_path = os.path.join(target_dir, filename)
            resize_fits_image(input_path, output_path)
            print(f"Resized {filename} and saved to {output_path}")


if __name__ == "__main__":
    # main()
    print_file_shape("data/256")
