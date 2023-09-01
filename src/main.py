import os
import time
import yaml
import shutil
import datetime
import tensorflow as tf
from data.dataset import Dataset
from training.train import Trainer
from models.generator import Generator
from data.data_loader import DataLoader
from models.discriminator import Discriminator


def load_config(config_path):
    """Loads the configuration file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_available_checkpoints():
    experiments_dir = './experiments'
    available_checkpoints = []
    
    # Check if experiments directory exists
    if os.path.exists(experiments_dir):
        for experiment in os.listdir(experiments_dir):
            checkpoint_dir = os.path.join(experiments_dir, experiment, 'training_checkpoints')
            if os.path.exists(checkpoint_dir):
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.index')]
                for checkpoint in checkpoints:
                    available_checkpoints.append((experiment, checkpoint))
    return available_checkpoints


def prompt_for_checkpoint(available_checkpoints):
    print("Would you like to resume training from a previous checkpoint?")
    choice = input("Enter 'yes' or 'no': ").strip().lower()
    if choice == 'yes':
        print("Available checkpoints:")
        for idx, (experiment, checkpoint) in enumerate(available_checkpoints, 1):
            if idx == 1 or available_checkpoints[idx-2][0] != experiment:
                print(f"{experiment}:")
            print(f"\t[{idx}] {checkpoint}")
        
        selected_checkpoint_idx = int(input("Enter the number of the checkpoint you want to resume from: "))
        return available_checkpoints[selected_checkpoint_idx - 1]
    return None



def check_for_data(experiment_dir):
    data_dir = os.path.join(experiment_dir, 'data')
    train_data_dir = os.path.join(data_dir, 'train')
    test_data_dir = os.path.join(data_dir, 'test')
    if os.path.exists(train_data_dir) and os.listdir(train_data_dir) and os.path.exists(test_data_dir) and os.listdir(test_data_dir):
        print("Data found in the selected experiment's datetime folder.")
        choice = input("Do you want to use this data? Enter 'yes' or 'no': ").strip().lower()
        return choice == 'yes'
    return False



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

    available_checkpoints = get_available_checkpoints()
    if available_checkpoints:
        selected_experiment, selected_checkpoint_file = prompt_for_checkpoint(available_checkpoints)
        use_existing_data = check_for_data(selected_experiment)
    else:
        print("No checkpoints available. Starting training from scratch.")
        use_existing_data = False

    if use_existing_data:
        train_directory = os.path.join('./experiments', selected_experiment, 'data', 'train')
        test_directory  = os.path.join('./experiments', selected_experiment, 'data', 'test')

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

    # Create data folder inside the experiment directory
    data_dir = os.path.join(experiment_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Create train and test folders inside the data directory
    train_data_dir = os.path.join(data_dir, 'train')
    test_data_dir  = os.path.join(data_dir, 'test')
    os.makedirs(train_data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)

    # Copy all files from train_directory to train_data_dir
    for file_name in os.listdir(train_directory):
        shutil.copy(os.path.join(train_directory, file_name), train_data_dir)

    # Copy all files from test_directory to test_data_dir
    for file_name in os.listdir(test_directory):
        shutil.copy(os.path.join(test_directory, file_name), test_data_dir)

    # Copy train_csv_path to train_data_dir and test_csv_path to test_data_dir
    shutil.copy(train_csv_path, train_data_dir)
    shutil.copy(test_csv_path, test_data_dir)

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

    # If resuming from a checkpoint, load the model weights
    if selected_experiment:
        checkpoint_path = os.path.join('./experiments', selected_experiment, 'training_checkpoints', selected_checkpoint_file)
        trainer.checkpoint.restore(checkpoint_path)

    # Train the model
    start_time = time.time()    
    trainer.fit(train_dataset, test_dataset, steps=STEPS, experiment_dir=experiment_dir)
    end_time = time.time()      

    # Calculate and display the total training time
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print("Training finished!")


if __name__ == "__main__":
    main()
