import os
import numpy as np

from astropy.io import fits

class PostProcessFITS:
    """
    A class to post-process FITS files. 
    Takes the 3D FITS files from the input directory, combines the 3 channels into a single channel, and denormalizes the data.
    This class is the inverse of the single_fits_pre.py file.

    Attributes:
        input_dir (str): Directory containing the input FITS files.
        output_dir (str): Directory to save the processed FITS files.
        data_type (int): Type of the FITS file (1 for magnetogram, 2 for intensity, 3 for divergence).

    Methods:
        denormalize_data(data, data_type): Denormalizes the data based on the data type.
        combine_channels(data): Combines the 3 channels into a single channel by averaging.
        process(): Processes the FITS files in the input directory and saves them in the output directory.
    """

    def __init__(self, input_dir, output_dir, data_type):
        """
        Initializes the PostProcessFITS class with the given input directory, output directory, and data type.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.data_type = data_type

    def denormalize_data(self, data, data_type):
        """
        Denormalizes the data based on the data type.

        Parameters:
            data (np.array): The data to be denormalized.
            data_type (int): Type of the data (1 for magnetogram, 2 for intensity, 3 for divergence).

        Returns:
            np.array: The denormalized data.
        """
        magnetogram_factor = 4000.0  # gauss
        intensity_factor = 50000.0  # ?
        divergence_factor = 100.0  # cm/s

        if data_type == 1:  # Magnetogram
            return data * magnetogram_factor
        elif data_type == 2:  # Intensity
            return data * intensity_factor
        elif data_type == 3:  # Divergence
            return data * divergence_factor
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def combine_channels(self, data):
        """
        Combines the 3 channels into a single channel by averaging.

        Parameters:
            data (np.array): The data with 3 channels.

        Returns:
            np.array: The combined data.
        """
        return np.mean(data, axis=-1)

    def process(self):
        """
        Processes the FITS files in the input directory and saves them in the output directory.
        """
        for file_name in os.listdir(self.input_dir):
            if file_name.endswith(".fits"):
                file_path = os.path.join(self.input_dir, file_name)

                # Read the 3D FITS file
                with fits.open(file_path) as hdul:
                    data = hdul[0].data

                # Combine the 3 channels into a single channel
                combined_data = self.combine_channels(data)

                # Denormalize the data
                denormalized_data = self.denormalize_data(combined_data, self.data_type)

                # Save the processed data
                output_path = os.path.join(self.output_dir, file_name)
                hdu = fits.PrimaryHDU(data=denormalized_data)
                hdu.writeto(output_path, overwrite=True)


if __name__ == "__main__":
    input_dir = input("Enter the input directory path (3D FITS files): ")
    output_dir = input("Enter the output directory path: ")
    data_type = int(
        input(
            "Enter the type of FITS file (1 for magnetogram, 2 for intensity, 3 for divergence): "
        )
    )
    post_processor = PostProcessFITS(input_dir, output_dir, data_type)
    post_processor.process()
