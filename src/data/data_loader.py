import tensorflow as tf
import os
import pathlib


class DataLoader:
    def __init__(self, dataset_name="facades"):
        self.dataset_name = dataset_name
        self._URL = f"http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{self.dataset_name}.tar.gz"
        self.PATH = self.download_and_extract_dataset()

    def download_and_extract_dataset(self):
        path_to_zip = tf.keras.utils.get_file(
            fname=f"{self.dataset_name}.tar.gz", origin=self._URL, extract=True
        )
        path_to_zip = pathlib.Path(path_to_zip)
        return path_to_zip.parent / self.dataset_name

    def load(self, image_file):
        image = tf.io.read_file(image_file)
        image = tf.io.decode_jpeg(image)

        w = tf.shape(image)[1]
        w = w // 2
        input_image = image[:, w:, :]
        real_image = image[:, :w, :]

        input_image = tf.cast(input_image, tf.float32)
        real_image = tf.cast(real_image, tf.float32)

        return input_image, real_image

    def load_image_train(self, image_file):
        input_image, real_image = self.load(image_file)
        input_image, real_image = self.normalize(input_image, real_image)
        return input_image, real_image

    def load_image_test(self, image_file):
        input_image, real_image = self.load(image_file)
        input_image, real_image = self.normalize(input_image, real_image)
        return input_image, real_image

    @staticmethod
    def normalize(input_image, real_image):
        input_image = (input_image / 127.5) - 1
        real_image = (real_image / 127.5) - 1
        return input_image, real_image
