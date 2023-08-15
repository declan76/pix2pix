import pathlib
import tensorflow as tf
from data.data_loader import DataLoader

class Dataset:
    def __init__(self, data_loader: DataLoader, buffer_size=400, batch_size=1):
        self.data_loader = data_loader
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    def create_dataset(self, type="train"):
        if type == "train":
            dataset_path = pathlib.Path("./data_set/train/*.fits")
        else:
            dataset_path = pathlib.Path("./data_set/test/*.fits")

        dataset = tf.data.Dataset.list_files(str(dataset_path))
        dataset = dataset.batch(2, drop_remainder=True)
        for batch in dataset.take(1):
            print(f"Shape after first batching: {batch.shape}")

        if type == "train":
            dataset = dataset.map(
                self.data_loader.load_image_train, num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            dataset = dataset.map(self.data_loader.load_image_test)

        for batch in dataset.take(1):
            print(f"Shape after mapping: {batch[0].shape}, {batch[1].shape}")

        dataset = dataset.batch(self.batch_size)
        for batch in dataset.take(1):
            print(f"Shape after second batching: {batch[0].shape}, {batch[1].shape}")

        return dataset



