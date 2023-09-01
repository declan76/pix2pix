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

def query_yes_no(prompt):
    while True:
        response = input(prompt + " (yes or no): ").strip().lower()
        if response in ["yes", "y", "1"]:
            return True
        elif response in ["no", "n", "0"]:
            return False
        else:
            print("Invalid input. Please enter yes/no, y/n, 1/0.")


def load_config(config_path):
    """Loads the configuration file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_available_checkpoints():
    experiments_dir       = './experiments'
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
    choice = query_yes_no("Would you like to resume training from a previous checkpoint?")
    if choice == True:
        print("Available checkpoints:")
        for idx, (experiment, checkpoint) in enumerate(available_checkpoints, 1):
            if idx == 1 or available_checkpoints[idx-2][0] != experiment:
                print(f"{experiment}:")
            print(f"\t[{idx}] {checkpoint}")
        
        selected_checkpoint_idx = int(input("Enter the number of the checkpoint you want to resume from: "))
        return available_checkpoints[selected_checkpoint_idx - 1]
    return None


def check_for_data(experiment_dir):
    base_dir = os.path.join('./experiments', experiment_dir, 'data')
    train_data_dir = os.path.join(base_dir, 'train')
    test_data_dir = os.path.join(base_dir, 'test')
    
    def has_data(directory):
        return os.path.exists(directory) and os.listdir(directory)
    
    if has_data(train_data_dir) and has_data(test_data_dir):
        print(f"Data found in {experiment_dir} folder.")
        return query_yes_no("Do you want to use this data?")
    return False



def main():
    # Load the configuration file
    config = load_config("config/hyperparameters.yaml")

    # Extract hyperparameters from the configuration
    BUFFER_SIZE = config["hyperparameters"]["BUFFER_SIZE"]
    BATCH_SIZE  = config["hyperparameters"]["BATCH_SIZE"]
    STEPS       = config["hyperparameters"]["STEPS"]
    TRAIN_RATIO = config["hyperparameters"]["TRAIN_RATIO"]

    # Extract paths from the configuration
    dataset_directory = config["paths"]["dataset_directory"]
    csv_path          = config["paths"]["csv_path"]
    train_directory   = config["paths"]["train_directory"]
    test_directory    = config["paths"]["test_directory"]
    train_csv_path    = config['paths']['train_csv_path']
    test_csv_path     = config['paths']['test_csv_path']

    # Get available checkpoints
    available_checkpoints = get_available_checkpoints()
    checkpoint_result     = None
    selected_experiment   = None

    # Prompt user to resume from a checkpoint if available
    if available_checkpoints:
        checkpoint_result = prompt_for_checkpoint(available_checkpoints)

    # Check if user wants to use existing data or start fresh
    if checkpoint_result:
        selected_experiment, selected_checkpoint_file = checkpoint_result
        use_existing_data = check_for_data(selected_experiment)
    else:
        message = "No checkpoints selected." if available_checkpoints else "No checkpoints available."
        print(f"{message} Starting training from scratch.")
        use_existing_data = False

    # Adjust paths if using existing data
    if use_existing_data:
        train_directory = os.path.join('./experiments', selected_experiment, 'data', 'train')
        test_directory  = os.path.join('./experiments', selected_experiment, 'data', 'test')
        train_csv_path  = os.path.join(train_directory, 'pairs.csv')
        test_csv_path   = os.path.join(test_directory, 'pairs.csv')
    else:
        # Split the data and save train/test pairs if not using existing data
        data_loader = DataLoader(dataset_directory, csv_path)
        train_pairs, test_pairs = data_loader.split_data(
            dataset_directory,
            train_directory,
            test_directory,
            csv_path,
            train_ratio=TRAIN_RATIO,
        )
        train_pairs.to_csv(train_csv_path, index=False, header=False)
        test_pairs.to_csv(test_csv_path, index=False, header=False)

    # Initialize data loaders and datasets
    train_data_loader = DataLoader(train_directory, train_csv_path)
    train_dataset     = Dataset(train_data_loader, BUFFER_SIZE, BATCH_SIZE).create_dataset()
    test_data_loader = DataLoader(test_directory, test_csv_path)
    test_dataset     = Dataset(test_data_loader, BUFFER_SIZE, BATCH_SIZE).create_dataset()

    # Initialize generator and discriminator models
    generator     = Generator()
    discriminator = Discriminator()
    generator.build_model()
    discriminator.build_model()

    # Create a unique directory for the current experiment
    experiment_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    experiment_dir       = os.path.join('./experiments', experiment_timestamp)
    os.makedirs(experiment_dir, exist_ok=True)

    # Save the current configuration to the experiment directory
    with open(os.path.join(experiment_dir, 'hyperparameters.yaml'), 'w') as file:
        yaml.dump(config, file)

    # Set paths for logging and checkpoints
    log_dir        = os.path.join(experiment_dir, 'logs', 'fit')
    checkpoint_dir = os.path.join(experiment_dir, 'training_checkpoints')

    # Initialize the trainer
    summary_writer    = tf.summary.create_file_writer(log_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    trainer           = Trainer(generator, discriminator, summary_writer, checkpoint_prefix)

    # Load model weights if resuming from a checkpoint
    if selected_experiment:
        checkpoint_path = os.path.join('./experiments', selected_experiment, 'training_checkpoints', selected_checkpoint_file)
        trainer.checkpoint.restore(checkpoint_path)

    # Train the model
    start_time = time.time()    
    trainer.fit(train_dataset, test_dataset, steps=STEPS, experiment_dir=experiment_dir)
    end_time = time.time()      

    # Display the total training time
    total_time       = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print("Training finished!")


if __name__ == "__main__":
    main()
