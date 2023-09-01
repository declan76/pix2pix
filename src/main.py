import os
import yaml
import datetime
import tensorflow as tf
from data.dataset import Dataset
from training.train import Trainer
from models.generator import Generator
from data.data_loader import DataLoader
from models.discriminator import Discriminator


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def main():
    # Load the configuration file
    config = load_config("config/hyperparameters.yaml")

    # Hyperparameters
    BUFFER_SIZE = config["hyperparameters"]["BUFFER_SIZE"]
    BATCH_SIZE  = config["hyperparameters"]["BATCH_SIZE"]
    STEPS       = config["hyperparameters"]["STEPS"]
    TRAIN_RATIO = config["hyperparameters"]["TRAIN_RATIO"]

    # Paths
    dataset_directory = config["paths"]["dataset_directory"]
    csv_path          = config["paths"]["csv_path"]
    train_directory   = config["paths"]["train_directory"]
    test_directory    = config["paths"]["test_directory"]
    train_csv_path    = config['paths']['train_csv_path']
    test_csv_path     = config['paths']['test_csv_path']

    data_loader = DataLoader(dataset_directory, csv_path)

    # Split the data and get the train and test pairs
    train_pairs, test_pairs = data_loader.split_data(
        dataset_directory,
        train_directory,
        test_directory,
        csv_path,
        train_ratio=TRAIN_RATIO,
    )

    # Save the training and testing pairs to separate CSV files
    train_pairs.to_csv(train_csv_path, index=False, header=False)
    test_pairs.to_csv(test_csv_path, index=False, header=False)

    # Create train_dataset using the training pairs CSV
    train_data_loader = DataLoader(train_directory, train_csv_path)
    train_dataset     = Dataset(train_data_loader, BUFFER_SIZE, BATCH_SIZE).create_dataset()

    # Create test_dataset using the testing pairs CSV
    test_data_loader = DataLoader(test_directory, test_csv_path)
    test_dataset     = Dataset(test_data_loader, BUFFER_SIZE, BATCH_SIZE).create_dataset()

    # Create the generator and discriminator models
    generator     = Generator()
    discriminator = Discriminator()
    generator.build_model()
    discriminator.build_model()

    # Create a unique directory for this experiment based on the current date-time
    experiment_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    experiment_dir       = os.path.join('./experiments', experiment_timestamp)
    os.makedirs(experiment_dir, exist_ok=True)

    # Copy the current configuration to the experiment directory
    with open(os.path.join(experiment_dir, 'hyperparameters.yaml'), 'w') as file:
        yaml.dump(config, file)

    # Update paths for logs, checkpoints, and generated images
    log_dir        = os.path.join(experiment_dir, 'logs', 'fit')
    checkpoint_dir = os.path.join(experiment_dir, 'training_checkpoints')

    # Create the summary writer and checkpoint manager
    summary_writer    = tf.summary.create_file_writer(log_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    trainer           = Trainer(generator, discriminator, summary_writer, checkpoint_prefix)

    # Train the model
    trainer.fit(train_dataset, test_dataset, steps=STEPS, experiment_dir=experiment_dir)


if __name__ == "__main__":
    main()
