import tensorflow as tf
import tensorflow_probability as tfp
import pathlib
from utils.fits_handler import read_fits, process_fits_data

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

        # Process the data to fit into the required channels
        processed_data = process_fits_data(data)

        # Convert to tensor
        input_image = tf.convert_to_tensor(processed_data, dtype=tf.float32)

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

        input_image, real_image = self.normalize(input_image, real_image)
        return input_image, real_image


    def load_image_test(self, image_files):
        image_file = image_files[0]
        next_image_file = image_files[1]

        input_image = self.load(image_file)
        real_image = self.load(next_image_file)

        if input_image is None or real_image is None:
            raise ValueError(f"Error: Input or real image is None for files: {image_files}")

        input_image, real_image = self.normalize(input_image, real_image)
        return input_image, real_image


    # @staticmethod
    # def normalize(input_image, real_image):
    #     # Normalize to [-1, 1]
    #     input_image = (input_image - tf.math.reduce_mean(input_image)) / (tf.math.reduce_max(input_image) - tf.math.reduce_min(input_image)) * 2 - 1
    #     real_image = (real_image - tf.math.reduce_mean(real_image)) / (tf.math.reduce_max(real_image) - tf.math.reduce_min(real_image)) * 2 - 1
    #     return input_image, real_image


    @staticmethod
    def normalize(input_image, real_image):
        # Normalize to [-1, 1]

        # Zero-mean normalization:
        # normalized_image=max_val−min_val/image−mean_val​×2−1        
        # input_image = (input_image - tf.math.reduce_mean(input_image)) / (tf.math.reduce_max(input_image) - tf.math.reduce_min(input_image)) * 2 - 1
        # real_image = (real_image - tf.math.reduce_mean(real_image)) / (tf.math.reduce_max(real_image) - tf.math.reduce_min(real_image)) * 2 - 1

        # Percentile clipping max normalisation: 
        # min_val_input = tf.reduce_min(input_image)
        # max_val_input = tf.reduce_max(input_image)
        # percentile_1_input = tfp.stats.percentile(input_image, 1.0)
        # percentile_99_input = tfp.stats.percentile(input_image, 99.0)
        # input_image = tf.clip_by_value(input_image, percentile_1_input, percentile_99_input)
        # input_image = (input_image - min_val_input) / (max_val_input - min_val_input) * 2 - 1
        # min_val_real = tf.reduce_min(real_image)
        # max_val_real = tf.reduce_max(real_image)
        # percentile_1_real = tfp.stats.percentile(real_image, 1.0)
        # percentile_99_real = tfp.stats.percentile(real_image, 99.0)
        # real_image = tf.clip_by_value(real_image, percentile_1_real, percentile_99_real)
        # real_image = (real_image - min_val_real) / (max_val_real - min_val_real) * 2 - 1

        # Min-max normalisation:
        # normalized_image=max_val−min_val/image−min_val​×2−1
        min_val = tf.reduce_min(input_image)
        max_val = tf.reduce_max(input_image)
        input_image = (input_image - min_val) / (max_val - min_val) * 2 - 1
        min_val = tf.reduce_min(real_image)
        max_val = tf.reduce_max(real_image)
        real_image = (real_image - min_val) / (max_val - min_val) * 2 - 1

        return input_image, real_image
