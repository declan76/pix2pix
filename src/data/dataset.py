import pathlib
import tensorflow as tf
from data.data_loader import DataLoader

class Dataset:
    def __init__(self, data_loader, buffer_size=400, batch_size=1):
        self.data_loader = data_loader
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    def create_dataset(self):
        num_pairs = len(self.data_loader.pairs)
        dataset = tf.data.Dataset.range(num_pairs)
        dataset = dataset.map(lambda idx: tf.py_function(self.data_loader.load_image_pair, [idx], [tf.float32, tf.float32]))
        dataset = dataset.batch(self.batch_size)
        return dataset
    