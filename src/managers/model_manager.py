import os
import sys
import yaml

from pix2pix.generator import Generator
from managers.file_manager import FileManager
from pix2pix.discriminator import Discriminator
from managers.user_input_manager import UserInputManager

class ModelManager:
    """
    The ModelManager class manages the loading and handling of model configurations, 
    checkpoints, and data directories. It provides methods to load hyperparameters, 
    retrieve available checkpoints, and prompt users for checkpoint selection.
    It is the parent class of the TrainingManager and EvaluationManager classes.
    """

    EXPERIMENTS_DIR      = "./experiments"
    HYPERPARAMETERS_PATH = "config/hyperparameters.yaml"

    def __init__(self):
        """
        Initializes the ModelManager by loading the configuration from the hyperparameters file.
        """
        self.config = self.load_config(self.HYPERPARAMETERS_PATH)

    def load_config(self, config_path):
        """
        Loads the configuration from the given path.

        Args:
            config_path (str): Path to the configuration file.

        Returns:
            dict: Loaded configuration.
        """
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def get_available_checkpoints(self):
        """
        Retrieves a list of available checkpoints from the experiments directory.

        Returns:
            list: List of tuples containing experiment name and checkpoint filename.
        """
        checkpoints     = []
        for experiment in os.listdir(self.EXPERIMENTS_DIR):
            checkpoint_dir = os.path.join(self.EXPERIMENTS_DIR, experiment, "training_checkpoints")
            if os.path.exists(checkpoint_dir):
                for f in os.listdir(checkpoint_dir):
                    if f.endswith(".index"):
                        checkpoints.append((experiment, f))
        return checkpoints

    def prompt_for_checkpoint(self):
        """
        Prompts the user to select a checkpoint either automatically from the experiments directory 
        or by manually providing a path.

        Returns:
            str: Path to the selected checkpoint.
        """
        choice = UserInputManager.query_yes_no("Would you like the system to automatically search the experiments directory for all available checkpoints?")
        if choice:
            available_checkpoints = self.get_available_checkpoints()
            if not available_checkpoints:
                print("No checkpoints found.")
                return None
            print("Available checkpoints:")
            for idx, (experiment, checkpoint) in enumerate(available_checkpoints, 1):
                print(f"[{idx}] {experiment}/training_checkpoints/{checkpoint}")
            selection = int(input("Please select a checkpoint by entering the number beside it: "))
            if 1 <= selection <= len(available_checkpoints):
                experiment, checkpoint = available_checkpoints[selection - 1]
                print(f"You've selected checkpoint {checkpoint} from {experiment}/training_checkpoints. Proceeding with this checkpoint..")
                checkpoint_path = os.path.join(
                    self.EXPERIMENTS_DIR,
                    experiment,
                    "training_checkpoints",
                    checkpoint.replace('.index', '')
                )
                return checkpoint_path
            else:
                print("Invalid selection.")
                sys.exit() 
        else:
            checkpoint_path = input("Please enter the full path to the checkpoint directory (./path/to/checkpoint/ckpt-n): ")
            print(f"You've selected checkpoint {os.path.basename(checkpoint_path)}. Proceeding with this checkpoint...")
            return checkpoint_path
    
    def get_data_directory(self, data_type):
        """
        Prompts the user for the path to the data directory of the specified type.

        Args:
            data_type (str): Type of data (e.g., "training" or "testing").

        Returns:
            str: Path to the data directory.

        Raises:
            ValueError: If no CSV file is found in the provided directory.
        """
        data_dir = input(f"Please enter the path to the {data_type} data directory: ")
        if FileManager.check_data_exists(data_dir):
            return data_dir
        else:
            raise ValueError(f"No CSV file found in the provided {data_type} directory.")
    
    @staticmethod
    def create_and_build_models():
        """
        Creates and builds the generator and discriminator models.

        Returns:
            tuple: Generator and discriminator models.
        """
        generator     = Generator()
        discriminator = Discriminator()
        generator.build_model()
        discriminator.build_model()
        return generator, discriminator