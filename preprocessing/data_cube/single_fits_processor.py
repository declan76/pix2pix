"""
Declan Keir 

29/08/2023
"""

import math
import os
from astropy.io import fits
import numpy as np


def distance_to_disk_centre(crlt_obs, crln_obs, crlt_ref, crln_ref):
    # input values in degrees

    # convert to radians
    crln_ref = crln_ref * np.pi / 180.0
    crlt_ref = crlt_ref * np.pi / 180.0
    crln_obs = crln_obs * np.pi / 180.0
    crlt_obs = crlt_obs * np.pi / 180.0

    dlon = abs(crln_obs - crln_ref)

    distance = math.acos(
        math.sin(crlt_obs) * math.sin(crlt_ref)
        + math.cos(crlt_obs) * math.cos(crlt_ref) * math.cos(dlon)
    )
    # distance=acos( cos(!dpi/2.-CRLT_OBS)*cos(!dpi/2.-CRLT_REF) + sin(!dpi/2.-CRLT_OBS)*sin(!dpi/2.-CRLT_REF)*cos(dlon)  )
    return distance  # in radians    (*180./!dpi for degrees)


def remove_2dplane(array):
    m = array.shape[0]  # assume it is square!

    X1, X2 = np.mgrid[:m, :m]
    X = np.hstack((np.reshape(X1, (m * m, 1)), np.reshape(X2, (m * m, 1))))
    X = np.hstack((np.ones((m * m, 1)), X))
    YY = np.reshape(array, (m * m, 1))
    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), YY)
    plane = np.reshape(np.dot(X, theta), (m, m))
    sub = array - plane

    # plt.imshow(array)
    # plt.show()
    # plt.imshow(sub)
    # plt.show()
    return sub


def read_fits(file_path):
    """Read a FITS file and return its data."""
    with fits.open(file_path) as hdul:
        data = hdul[1].data
        if data is None:
            raise ValueError(f"Error: No data in FITS file: {file_path}")
    return data


def duplicate_data(data, channels=3):
    """Duplicate the FITS data to fit into the required channels."""
    duplicated_data = np.stack([data] * channels, axis=-1)
    return duplicated_data


def normalize_data(data, data_type, header=None):
    magnetogram_factor  = 4000.0    # gauss
    intensity_factor    = 50000.0   # ?
    divergence_factor   = 100.0     # cm/s

    if data_type == 1:      # Magnetogram
        if header:
            theta = distance_to_disk_centre(
                header["CRLT_OBS"],
                header["CRLN_OBS"],
                header["CRLT_REF"],
                header["CRLN_REF"],
            )
            data = data / math.cos(theta)
        return data / magnetogram_factor
    elif data_type == 2:    # Intensity
        data = remove_2dplane(data)
        return data / intensity_factor
    elif data_type == 3:    # Divergence
        return data / divergence_factor
    else:
        raise ValueError(f"Unknown data type: {data_type}")


def resize_data(data, target_shape=(256, 256)):
    """Resize the data using either cropping or interpolation."""
    cropped_data = data[: target_shape[0], : target_shape[1], :]
    return cropped_data
    

def process_directory(input_dir, output_dir, data_type):
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".fits"):
            file_path = os.path.join(input_dir, file_name)

            # Read the FITS file
            with fits.open(file_path) as hdul:
                data = hdul[1].data
                header = hdul[1].header

            # Duplicate the data
            duplicated_data = duplicate_data(data)

            # Normalize the data
            normalized_data = normalize_data(duplicated_data, data_type, header)

            # Ensure data is within range [-1, 1]
            normalized_data = np.clip(normalized_data, -1, 1)

            # Resize the data
            resized_data = resize_data(normalized_data)

            # Save the processed data
            output_path = os.path.join(output_dir, file_name)
            hdu = fits.PrimaryHDU(data=resized_data)
            hdu.writeto(output_path, overwrite=True)


if __name__ == "__main__":
    input_dir = input("Enter the input directory path: ")
    output_dir = input("Enter the output directory path: ")
    data_type = int(input("Enter the type of FITS file (1 for magnetogram, 2 for intensity, 3 for divergence): "))
    process_directory(input_dir, output_dir, data_type)
