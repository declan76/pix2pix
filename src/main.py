import os
import sys
import time
import yaml
import shutil
import datetime
import traceback
import tensorflow as tf

from data.dataset import Dataset
from pix2pix.train import Trainer
from utils.pdf_writer import PDFWriter
from data.data_loader import DataLoader
from pix2pix.generator import Generator
from utils.visualise import generate_images
from pix2pix.discriminator import Discriminator


class UserInputManager:
    @staticmethod
    def query_yes_no(prompt):
        while True:
            response = input(prompt + " (yes/no): ").strip().lower()
            if response in ["yes", "y", "1"]:
                return True
            elif response in ["no", "n", "0"]:
                return False
            else:
                print("Invalid input. Please enter yes/no, y/n, 1/0.")

    @staticmethod
    def get_action():
        action = input("Do you want to train or evaluate? (t/e): ").strip().lower()
        while action not in ["t", "e"]:
            print("Invalid input. Please enter t or e.")
            action = input("Do you want to train or evaluate? (t/e): ").strip().lower()
        return action



class FileManager:
    @staticmethod
    def check_data_exists(directory):
        return os.path.exists(directory) and any(file.endswith(".csv") for file in os.listdir(directory))

    @staticmethod
    def copy_data_to_folder(source_dir, dest_dir):
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        shutil.copytree(source_dir, dest_dir)



class ModelManager:
    EXPERIMENTS_DIR      = "./experiments"
    HYPERPARAMETERS_PATH = "config/hyperparameters.yaml"

    def __init__(self):
        self.config = self.load_config(self.HYPERPARAMETERS_PATH)

    def load_config(self, config_path):
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def get_available_checkpoints(self):
        checkpoints     = []
        for experiment in os.listdir(self.EXPERIMENTS_DIR):
            checkpoint_dir = os.path.join(self.EXPERIMENTS_DIR, experiment, "training_checkpoints")
            if os.path.exists(checkpoint_dir):
                for f in os.listdir(checkpoint_dir):
                    if f.endswith(".index"):
                        checkpoints.append((experiment, f))
        return checkpoints

    def prompt_for_checkpoint(self):
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
        data_dir = input(f"Please enter the path to the {data_type} data directory: ")
        if FileManager.check_data_exists(data_dir):
            return data_dir
        else:
            raise ValueError(f"No CSV file found in the provided {data_type} directory.")
    
    @staticmethod
    def create_and_build_models():
        generator     = Generator()
        discriminator = Discriminator()
        generator.build_model()
        discriminator.build_model()
        return generator, discriminator




class TrainingManager(ModelManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def current_experiment_timestamp(self):
        return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def create_experiment_directory(self):
        experiment_dir = os.path.join(self.EXPERIMENTS_DIR, self.current_experiment_timestamp)
        os.makedirs(experiment_dir, exist_ok=True)
        return experiment_dir

    def train_model(self, train_csv_path, test_csv_path, experiment_dir, checkpoint_path=None):
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

        total_time = end_time - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print("Training finished!")


    def orchestrate_training(self):
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



class EvaluationManager(ModelManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_default_evaluation_path(self, checkpoint_path):
        base_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        return os.path.join(base_dir, "evaluation")

    def get_final_save_path(self, checkpoint_path):
        timestamp       = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        checkpoint_name = os.path.basename(checkpoint_path)
        folder_name     = f"{timestamp}_{checkpoint_name}"
        return os.path.join(self.get_default_evaluation_path(checkpoint_path), folder_name), timestamp

    def save_evaluation_results(self, final_save_path, mse_values):
        os.makedirs(final_save_path, exist_ok=True)

        mse_file_path = os.path.join(final_save_path, "MSE.CSV")
        with open(mse_file_path, "w") as file:
            for filename, mse in mse_values.items():
                file.write(f"{filename},{mse}\n")

        print(f"Evaluation results saved to {mse_file_path}")


    def evaluate_model(self, test_csv_path, checkpoint_path, save_images_path):
        # Ensure the checkpoint file exists
        if not os.path.exists(checkpoint_path + ".index"):
            raise ValueError(f"Checkpoint file {checkpoint_path}.index does not exist.")
        
        test_data_loader = DataLoader(test_csv_path, test_csv_path)
        test_dataset     = Dataset(test_data_loader, self.config["hyperparameters"]["BUFFER_SIZE"], self.config["hyperparameters"]["BATCH_SIZE"]).create_dataset()

        generator, discriminator = self.create_and_build_models()
       
        trainer = Trainer(generator, discriminator, None, None)
        status  = trainer.checkpoint.restore(checkpoint_path)
        status.expect_partial()

        mse_values = {}
        images     = []  # To store the paths of the top 3 and worst MSE images
        for idx, (file_name, input, target) in enumerate(test_dataset):
            prediction                                       = generator.model(input, training=True)
            mse_loss                                         = tf.keras.losses.MeanSquaredError()(target, prediction)
            mse_values[file_name.numpy()[0].decode('utf-8')] = mse_loss.numpy()
            print(f"MSE for test file {file_name.numpy()[0].decode('utf-8')}: {mse_loss.numpy()}")

            # Save generated images if path is provided
            if save_images_path:
                taget_name = file_name.numpy()[0].decode('utf-8')
                image_name = f"{taget_name}_predicted.png"
                image_path = os.path.join(save_images_path, image_name)
                tf.keras.preprocessing.image.save_img(image_path, prediction[0])

                generate_images(generator, input, target, taget_name, save_images_path, mode='eval')

        # Create temp folder to store images for PDF report
        os.makedirs("temp", exist_ok=True)
        
        # Identify top 3 and worst MSE files
        sorted_mse_items = sorted(mse_values.items(), key=lambda x: x[1])
        top_3_files      = sorted_mse_items[:3]
        worst_file       = sorted_mse_items[-1]

        # Generate images for these files
        for rank, (file_name, mse) in enumerate(top_3_files + [worst_file]):
            _, input, target = next(filter(lambda x: x[0].numpy()[0].decode('utf-8') == file_name, test_dataset))
            image_path       = generate_images(generator, input, target, file_name, "temp", mode='eval')
            images.append((rank, image_path))

        avg_mse = sum(mse_values.values()) / len(mse_values)
        print(f"Average MSE on test data: {avg_mse}")
        return mse_values, images  


    def orchestrate_evaluation(self):
        try:
            checkpoint_path = self.prompt_for_checkpoint()
            test_path       = self.get_data_directory("testing")
            test_csv_path   = os.path.join(test_path, "pairs.csv")
            if not os.path.exists(test_csv_path):
                raise ValueError(f"No CSV file found in the provided test directory.")

            final_save_path, timestamp = self.get_final_save_path(checkpoint_path)

            save_images = UserInputManager.query_yes_no("Do you want to save the generated images?")
            if save_images:
                save_images_path = os.path.join(final_save_path, "generated_images")
                os.makedirs(save_images_path, exist_ok=True)
            else:
                save_images_path = None

            FileManager.copy_data_to_folder(test_path, os.path.join(final_save_path, "data", "test"))    

            mse_values, images = self.evaluate_model(test_csv_path, checkpoint_path, save_images_path)
            self.save_evaluation_results(final_save_path, mse_values)

            # Generate PDF report with top images
            PDFWriter.generate_pdf_report(checkpoint_path, timestamp, mse_values, final_save_path, images)

        except Exception as e:
            print("-" * 50)
            print(f"An error occurred: {e}")
            print("Traceback:")
            traceback.print_exc()


def main():
    action = UserInputManager.get_action()

    if action == "t":
        training_manager = TrainingManager()
        training_manager.orchestrate_training()
    elif action == "e":
        evaluator_manager = EvaluationManager()
        evaluator_manager.orchestrate_evaluation()
        

if __name__ == "__main__":
    main()
