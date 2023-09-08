import shutil
import pathlib
import pandas as pd
import tensorflow as tf
from utils.fits_handler import read_fits

class DataLoader:
    def __init__(self, dataset_directory, csv_path):
        # Get the parent directory of the CSV file
        self.dataset_directory = pathlib.Path(dataset_directory).parent  
        # Read the CSV file into a DataFrame
        self.pairs = pd.read_csv(csv_path)
        # Shuffle the rows of the DataFrame
        self.pairs = self.pairs.sample(frac=1).reset_index(drop=True)

    def load(self, image_file):
        if not image_file.endswith('.fits'):
            image_file += '.fits'
        image_file_path = self.dataset_directory / image_file
        data = tf.py_function(func=self._load_image, inp=[str(image_file_path)], Tout=tf.float32)
        return tf.convert_to_tensor(data, dtype=tf.float32)

    def _load_image(self, image_file_path_tensor):
        image_file_path = image_file_path_tensor.numpy().decode('utf-8')
        data            = read_fits(image_file_path)
        if data is None:
            print(f"Data is None for file: {image_file_path}")
        return data

    def load_image_pair(self, index_tensor):
        index = index_tensor.numpy()         # Convert tensor to numpy array
        input_image_name, real_image_name = self.pairs.iloc[int(index)]  # Convert numpy array to integer
        input_image = self.load(input_image_name)
        real_image  = self.load(real_image_name)
        return input_image_name, input_image, real_image


    @staticmethod
    def split_data(dataset_path, train_path, test_path, csv_path, train_ratio=0.8):
        dataset_path_obj = pathlib.Path(dataset_path)
        train_path_obj   = pathlib.Path(train_path)
        test_path_obj    = pathlib.Path(test_path)

        # Function to clear the contents of a directory
        def clear_directory(directory):
            for item in directory.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

        # Ensure train and test directories exist
        train_path_obj.mkdir(parents=True, exist_ok=True)
        test_path_obj.mkdir(parents=True, exist_ok=True)

        # Clear the contents of train and test directories if they exist and have data
        if any(train_path_obj.iterdir()):
            clear_directory(train_path_obj)
        if any(test_path_obj.iterdir()):
            clear_directory(test_path_obj)

        # Read the CSV file into a DataFrame and shuffle the pairs
        pairs = pd.read_csv(csv_path, header=None)  
        pairs = pairs.sample(frac=1).reset_index(drop=True)

        # Split the shuffled pairs DataFrame into training and testing datasets
        train_size  = int(train_ratio * len(pairs))
        train_pairs = pairs.iloc[:train_size]
        test_pairs  = pairs.iloc[train_size:]

        # Copy the files based on the split pairs to train and test directories
        for _, row in train_pairs.iterrows():
            shutil.copy(dataset_path_obj / row.iloc[0], train_path_obj / row.iloc[0])   # Use iloc
            shutil.copy(dataset_path_obj / row.iloc[1], train_path_obj / row.iloc[1])   # Use iloc

        for _, row in test_pairs.iterrows():
            shutil.copy(dataset_path_obj / row.iloc[0], test_path_obj / row.iloc[0])    # Use iloc
            shutil.copy(dataset_path_obj / row.iloc[1], test_path_obj / row.iloc[1])    # Use iloc

        return train_pairs, test_pairs


