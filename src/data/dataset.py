import tensorflow as tf
from .data_loader import DataLoader


class Dataset:
    def __init__(self, data_loader: DataLoader, buffer_size=400, batch_size=1):
        self.data_loader = data_loader
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    def create_dataset(self, type="train"):
        try:
            dataset = tf.data.Dataset.list_files(
                str(self.data_loader.PATH / f"{type}/*.jpg")
            )
        except tf.errors.InvalidArgumentError:
            dataset = tf.data.Dataset.list_files(
                str(self.data_loader.PATH / "val/*.jpg")
            )

        if type == "train":
            dataset = dataset.map(
                self.data_loader.load_image_train, num_parallel_calls=tf.data.AUTOTUNE
            )
            dataset = dataset.shuffle(self.buffer_size)
        else:
            dataset = dataset.map(self.data_loader.load_image_test)

        dataset = dataset.batch(self.batch_size)

        return dataset
