import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from astropy.io import fits

def read_fits(file_path, file_path_str=None):
    """
    Read a FITS file and return its data.
    """
    if file_path_str is None:
        file_path_str = file_path if isinstance(file_path, str) else file_path.numpy().decode('utf-8')
    with fits.open(file_path_str) as hdul:
        data = hdul[0].data
        if data is None:
            raise ValueError(f"Error: No data in FITS file: {file_path_str}")
    
    return data

def display_fits_image(file_path_str):
    with fits.open(file_path_str) as hdul:
        for i, hdu in enumerate(hdul):
            data = hdu.data
            if data is not None:
                plt.figure(figsize=(10, 10))
                plt.imshow(data, cmap='gray')
                plt.title(f"File: {file_path_str.name} | HDU {i}")
                plt.colorbar()
                # plt.show() 
                plt.savefig(f"./data_set/images/{file_path_str.name}_{i}.png")

