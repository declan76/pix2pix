import pathlib
import pandas as pd
import tensorflow as tf

from utils.image_processor import ImageProcessor

class DataLoader:
    """
    DataLoader class for loading and processing FITS image files.
    """
    
    def __init__(self, dataset_directory, csv_path):
        """
        Initializes the DataLoader with the dataset directory and CSV path.
        
        Args:
        - dataset_directory (str): Path to the directory containing the dataset.
        - csv_path (str): Path to the CSV file containing image pairs.
        """
        self.dataset_directory = pathlib.Path(dataset_directory).parent
        self.pairs             = pd.read_csv(csv_path)
        self.pairs             = self.pairs.sample(frac=1).reset_index(drop=True)

    def load(self, image_file):
        """
        Loads the image file and returns its data as a tensor.
        
        Args:
        - image_file (str): Name of the image file to load.
        
        Returns:
        - tf.Tensor: Tensor representation of the image data.
        """
        # Ensure the image file has the correct extension
        if not image_file.endswith('.fits'):
            image_file += '.fits'
        image_file_path = self.dataset_directory / image_file
        data = tf.py_function(func=self._load_image, inp=[str(image_file_path)], Tout=tf.float32)
        return tf.convert_to_tensor(data, dtype=tf.float32)

    def _load_image(self, image_file_path_tensor):
        """
        Helper function to load the image data from the given file path tensor.
        
        Args:
        - image_file_path_tensor (tf.Tensor): Tensor containing the path to the image file.
        
        Returns:
        - data: Data read from the FITS image file.
        """
        image_file_path = image_file_path_tensor.numpy().decode('utf-8')
        data            = ImageProcessor().read_fits(image_file_path)
        if data is None:
            print(f"Data is None for file: {image_file_path}")
        return data

    def load_image_pair(self, index_tensor):
        """
        Loads a pair of images based on the given index tensor.
        
        Args:
        - index_tensor (tf.Tensor): Tensor containing the index of the image pair to load.
        
        Returns:
        - tuple: Tuple containing the names and tensor representations of the input and real images.
        """
        index                             = index_tensor.numpy()         # Convert tensor to numpy array
        input_image_name, real_image_name = self.pairs.iloc[int(index)]  # Convert numpy array to integer
        input_image                       = self.load(input_image_name)
        real_image                        = self.load(real_image_name)
        return input_image_name, input_image, real_image
