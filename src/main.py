import os
import tensorflow as tf
import datetime
from data.data_loader import DataLoader
from data.dataset import Dataset
from models.generator import Generator
from models.discriminator import Discriminator
from training.train import Trainer
from evaluation.visualise import generate_images


def main():
    # Set the parameters
    BUFFER_SIZE = 400
    BATCH_SIZE = 1
    STEPS = 200000  # Set the number of steps you want for training

    # Create the data loader and dataset
    data_loader = DataLoader()
    train_dataset = Dataset(data_loader, BUFFER_SIZE, BATCH_SIZE).create_dataset(
        "train"
    )
    test_dataset = Dataset(data_loader, BUFFER_SIZE, BATCH_SIZE).create_dataset("test")

    # Create the generator and discriminator
    generator = Generator()
    generator.build_model()
    discriminator = Discriminator()
    discriminator.build_model()

    # Create the trainer
    log_dir = "./logs/"
    summary_writer = tf.summary.create_file_writer(
        log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    checkpoint_dir = "./training_checkpoints"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    trainer = Trainer(generator, discriminator, summary_writer, checkpoint_prefix)

    # Run the training
    trainer.fit(train_dataset, test_dataset, steps=STEPS)


if __name__ == "__main__":
    main()
