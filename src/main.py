import os
import pathlib
import tensorflow as tf
import datetime
from data.data_loader import DataLoader
from data.dataset import Dataset
from models.generator import Generator
from models.discriminator import Discriminator
from training.train import Trainer
from evaluation.visualise import generate_images


def main():
    BUFFER_SIZE = 400
    BATCH_SIZE = 1
    STEPS = 200000
    TRAIN_RATIO = 0.8

    csv_path = "data_set/data/single_fits_processor-mf/pairs.csv"
    dataset_directory = "data_set/data/single_fits_processor-mf/output"
    train_directory = "data_set/train"
    test_directory = "data_set/test"

    data_loader = DataLoader(dataset_directory, csv_path)

    # Split the data and get the train and test pairs
    train_pairs, test_pairs = data_loader.split_data(
        dataset_directory, train_directory, test_directory, csv_path, train_ratio=TRAIN_RATIO
    )

    # Save the training and testing pairs to separate CSV files
    train_csv_path = "data_set/train/pairs.csv"
    test_csv_path = "data_set/test/pairs.csv"
    train_pairs.to_csv(train_csv_path, index=False, header=False)  
    test_pairs.to_csv(test_csv_path, index=False, header=False)   

    # Create train_dataset using the training pairs CSV
    train_data_loader = DataLoader(train_directory, train_csv_path)
    train_dataset = Dataset(train_data_loader, BUFFER_SIZE, BATCH_SIZE).create_dataset()

    # Create test_dataset using the testing pairs CSV
    test_data_loader = DataLoader(test_directory, test_csv_path)
    test_dataset = Dataset(test_data_loader, BUFFER_SIZE, BATCH_SIZE).create_dataset()

    generator = Generator()
    generator.build_model()
    discriminator = Discriminator()
    discriminator.build_model()

    log_dir = "./logs/"
    summary_writer = tf.summary.create_file_writer(
        log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    checkpoint_dir = "./training_checkpoints"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    trainer = Trainer(generator, discriminator, summary_writer, checkpoint_prefix)

    trainer.fit(train_dataset, test_dataset, steps=STEPS)


if __name__ == "__main__":
    main()
