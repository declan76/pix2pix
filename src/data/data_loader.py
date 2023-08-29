import tensorflow as tf
import tensorflow_probability as tfp
import pathlib
from utils.fits_handler import read_fits

class DataLoader:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.PATH = pathlib.Path(dataset_name)
    
    @staticmethod
    def split_data(dataset_path, train_path, test_path, train_ratio=0.8):
        dataset_path_obj = pathlib.Path(dataset_path)
        train_path_obj = pathlib.Path(train_path)
        test_path_obj = pathlib.Path(test_path)
        
        all_files = sorted(list(dataset_path_obj.glob("*.fits")))
        total_files = len(all_files)
        
        # Split the dataset into training and testing sets
        train_size = int(train_ratio * total_files)
        train_files = all_files[:train_size]
        test_files = all_files[train_size:]
        
        # Move the files to the respective directories
        for file in train_files:
            file.rename(train_path_obj / file.name)
        for file in test_files:
            file.rename(test_path_obj / file.name)

    def load(self, image_file):
        # Directly load the image using the provided file path
        image_file_str = tf.strings.as_string(image_file, precision=-1, scientific=False, shortest=False, width=-1)

        # Read the image using tf.py_function
        data = tf.py_function(
            func=self._load_image, 
            inp=[image_file_str], 
            Tout=tf.float32
        )

        # Convert to tensor
        input_image = tf.convert_to_tensor(data, dtype=tf.float32)

        return input_image

    def _load_image(self, image_file_str):
        data = read_fits(image_file_str)
        
        if data is None:
            print(f"Data is None for file: {image_file_str}")
        
        return data

    def load_image_train(self, image_files):
        image_file = image_files[0]
        next_image_file = image_files[1]

        input_image = self.load(image_file)
        real_image = self.load(next_image_file)

        if input_image is None or real_image is None:
            raise ValueError(f"Error: Input or real image is None for files: {image_files}")
        
        return input_image, real_image

    def load_image_test(self, image_files):
        image_file = image_files[0]
        next_image_file = image_files[1]

        input_image = self.load(image_file)
        real_image = self.load(next_image_file)

        if input_image is None or real_image is None:
            raise ValueError(f"Error: Input or real image is None for files: {image_files}")
        
        return input_image, real_image
    
