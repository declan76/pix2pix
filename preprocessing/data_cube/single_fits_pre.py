import os
import math
import numpy as np
from astropy.io import fits


def distance_to_disk_centre(crlt_obs, crln_obs, crlt_ref, crln_ref):
    """
    Calculate the distance to the disk center.

    Parameters:
        crlt_obs, crln_obs (float): Observer's latitude and longitude in degrees.
        crlt_ref, crln_ref (float): Reference latitude and longitude in degrees.

    Returns:
        float: Distance to the disk center in radians.
    """
    # Convert to radians
    crln_ref, crlt_ref, crln_obs, crlt_obs = [
        angle * np.pi / 180.0 for angle in [crln_ref, crlt_ref, crln_obs, crlt_obs]
    ]

    dlon = abs(crln_obs - crln_ref)

    distance = math.acos(
        math.sin(crlt_obs) * math.sin(crlt_ref)
        + math.cos(crlt_obs) * math.cos(crlt_ref) * math.cos(dlon)
    )
    return distance


def remove_2dplane(array):
    """
    Remove a 2D plane from the array.

    Parameters:
        array (np.array): Input 2D array.

    Returns:
        np.array: Array after removing the 2D plane.
    """
    m = array.shape[0]  # Assume it is square!

    X1, X2 = np.mgrid[:m, :m]
    X = np.hstack((np.reshape(X1, (m * m, 1)), np.reshape(X2, (m * m, 1))))
    X = np.hstack((np.ones((m * m, 1)), X))
    YY = np.reshape(array, (m * m, 1))
    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), YY)
    plane = np.reshape(np.dot(X, theta), (m, m))
    sub = array - plane

    return sub


def read_fits(file_path):
    """
    Read a FITS file and return its data.

    Parameters:
        file_path (str): Path to the FITS file.

    Returns:
        np.array: Data from the FITS file.
    """
    with fits.open(file_path) as hdul:
        data = hdul[1].data
        if data is None:
            print(50*"-")
            print(f"Error: No data in FITS file: {file_path}")
            raise ValueError
    return data


def duplicate_data(data, channels=3):
    """
    Duplicate the FITS data to fit into the required channels.

    Parameters:
        data (np.array): Input data.
        channels (int, optional): Number of channels. Defaults to 3.

    Returns:
        np.array: Duplicated data.
    """
    return np.stack([data] * channels, axis=-1)


def normalize_data(data, data_type, header=None):
    """
    Normalize the data based on the data type.

    Parameters:
        data (np.array): Input data.
        data_type (int): Type of the data (1 for magnetogram, 2 for intensity, 3 for divergence).
        header (dict, optional): Header information. Defaults to None.

    Returns:
        np.array: Normalized data.
    """
    magnetogram_factor = 4000.0  # Gauss
    intensity_factor = 50000.0
    divergence_factor = 100.0  # cm/s

    if data_type == 1:  # Magnetogram
        if header:
            theta = distance_to_disk_centre(
                header["CRLT_OBS"],
                header["CRLN_OBS"],
                header["CRLT_REF"],
                header["CRLN_REF"],
            )
            data = data / math.cos(theta)
        return data / magnetogram_factor
    elif data_type == 2:  # Intensity
        return remove_2dplane(data) / intensity_factor
    elif data_type == 3:  # Divergence
        return data / divergence_factor
    else:
        print(50*"-")
        print(f"Unknown data type: {data_type}")
        raise ValueError


def resize_data(data, target_shape=(256, 256)):
    """
    Resize the data using either cropping or interpolation.

    Parameters:
        data (np.array): Input data.
        target_shape (tuple, optional): Target shape for resizing. Defaults to (256, 256).

    Returns:
        np.array: Resized data.
    """
    return data[: target_shape[0], : target_shape[1], :]


def process_directory(input_dir, output_dir, data_type):
    """
    Process FITS files in the input directory and save them in the output directory.

    Parameters:
        input_dir (str): Directory containing the input FITS files.
        output_dir (str): Directory to save the processed FITS files.
        data_type (int): Type of the FITS file (1 for magnetogram, 2 for intensity, 3 for divergence).
    """
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
