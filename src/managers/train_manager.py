import os
import time
import yaml
import datetime
import traceback
import tensorflow as tf

from data.dataset import Dataset
from pix2pix.train import Trainer
from data.data_loader import DataLoader
from managers.file_manager import FileManager
from managers.model_manager import ModelManager
from managers.user_input_manager import UserInputManager

class TrainingManager(ModelManager):
    """
    The TrainingManager class is responsible for orchestrating the training process of the model.
    It inherits from the ModelManager class.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the TrainingManager class.
        """
        super().__init__(*args, **kwargs)

    @property
    def current_experiment_timestamp(self):
        """
        Returns the current timestamp in the format "YYYY-MM-DD_HH-MM-SS".
        """
        return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def create_experiment_directory(self):
        """
        Creates a new directory for the current experiment based on the current timestamp.
        Returns the path to the created directory.
        """
        experiment_dir = os.path.join(self.EXPERIMENTS_DIR, self.current_experiment_timestamp)
        os.makedirs(experiment_dir, exist_ok=True)
        return experiment_dir

    def train_model(self, train_csv_path, test_csv_path, experiment_dir, checkpoint_path=None):
        """
        Trains the model using the provided training and testing data.
        
        Args:
        - train_csv_path (str): Path to the training CSV file.
        - test_csv_path (str): Path to the testing CSV file.
        - experiment_dir (str): Path to the directory where the experiment data will be stored.
        - checkpoint_path (str, optional): Path to a checkpoint to resume training from. Defaults to None.
        """
        train_data_loader = DataLoader(train_csv_path, train_csv_path)
        train_dataset     = Dataset(train_data_loader, self.config["hyperparameters"]["BUFFER_SIZE"], self.config["hyperparameters"]["BATCH_SIZE"]).create_dataset()

        test_data_loader = DataLoader(test_csv_path, test_csv_path)
        test_dataset     = Dataset(test_data_loader, self.config["hyperparameters"]["BUFFER_SIZE"], self.config["hyperparameters"]["BATCH_SIZE"]).create_dataset()

        generator, discriminator = self.create_and_build_models()

        with open(os.path.join(experiment_dir, "hyperparameters.yaml"), "w") as file:
            yaml.dump(self.config, file)

        log_dir        = os.path.join(experiment_dir, "logs", "fit")
        checkpoint_dir = os.path.join(experiment_dir, "training_checkpoints")

        summary_writer    = tf.summary.create_file_writer(log_dir)
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        trainer           = Trainer(generator, discriminator, summary_writer, checkpoint_prefix)

        if checkpoint_path:
            trainer.checkpoint.restore(checkpoint_path)

        start_time = time.time()
        trainer.fit(train_dataset, test_dataset, steps=self.config["hyperparameters"]["STEPS"], experiment_dir=experiment_dir, save_freq=self.config["hyperparameters"]["SAVE_FREQ"])        
        end_time = time.time()

        total_time       = end_time - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print("Training finished!")


    def orchestrate_training(self):
        """
        Orchestrates the entire training process, including data preparation, model training, and error handling.
        """
        try:
            experiment_dir = self.create_experiment_directory()
            
            training_data_dir = self.get_data_directory("training")
            FileManager.copy_data_to_folder(training_data_dir, os.path.join(experiment_dir, "data", "train"))
            train_csv_path = os.path.join(experiment_dir, "data", "train", "pairs.csv")

            test_data_dir = self.get_data_directory("testing")
            FileManager.copy_data_to_folder(test_data_dir, os.path.join(experiment_dir, "data", "test"))
            test_csv_path = os.path.join(experiment_dir, "data", "test", "pairs.csv")

            checkpoint_path = None

            if UserInputManager.query_yes_no("Would you like to use a checkpoint from a previous run?"):
                checkpoint_path = self.prompt_for_checkpoint()
                
            self.train_model(train_csv_path, test_csv_path, experiment_dir, checkpoint_path)

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            traceback.print_exc()