import os
import time
import yaml
import shutil
import datetime
import traceback
import tensorflow as tf
from data.dataset import Dataset
from pix2pix.train import Trainer
from data.data_loader import DataLoader
from pix2pix.generator import Generator
from pix2pix.discriminator import Discriminator


def query_yes_no(prompt):
    while True:
        response = input(prompt + " (yes/no): ").strip().lower()
        if response in ["yes", "y", "1"]:
            return True
        elif response in ["no", "n", "0"]:
            return False
        else:
            print("Invalid input. Please enter yes/no, y/n, 1/0.")



class ModelManager:
    EXPERIMENTS_DIR      = "./experiments"
    HYPERPARAMETERS_PATH = "config/hyperparameters.yaml"

    def __init__(self, base_dir="data_set/data"):
        self.base_dir = base_dir
        self.config   = self.load_config(self.HYPERPARAMETERS_PATH)

    def load_config(self, config_path):
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def get_available_checkpoints(self):
        experiments_dir = "./experiments"
        checkpoints     = []
        for experiment in os.listdir(experiments_dir):
            checkpoint_dir = os.path.join(experiments_dir, experiment, "training_checkpoints")
            if os.path.exists(checkpoint_dir):
                for f in os.listdir(checkpoint_dir):
                    if f.endswith(".index"):
                        checkpoints.append((experiment, f))
        return checkpoints

    def prompt_for_checkpoint(self, available_checkpoints):
        print("Available checkpoints:")
        for idx, (experiment, checkpoint) in enumerate(available_checkpoints, 1):
            print(f"{idx}. {experiment}: {checkpoint}")
        return available_checkpoints[int(input("Enter the number of the checkpoint you want to resume from: ")) - 1]

    def check_for_data(self, experiment_dir):
        base_dir = os.path.join("./experiments", experiment_dir, "data")
        has_data = lambda directory: os.path.exists(directory) and os.listdir(directory)
        if all(map(has_data, [os.path.join(base_dir, d) for d in ["train", "test"]])):
            print(f"Data found in {experiment_dir} folder.")
            return query_yes_no("Do you want to use this data?")
        print("No data found in the experiment directory.")
        return False



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

    def handle_checkpoint_data(self, checkpoint_result, experiment_dir):
        if not checkpoint_result:
            return False, None, None

        selected_experiment, selected_checkpoint_file = checkpoint_result
        selected_data_directory = os.path.join("./experiments", selected_experiment, "data")
        if self.is_data_already_split(selected_data_directory):
            if query_yes_no(f"Data found in {selected_experiment} directory. Would you like to use it?"):
                self.copy_data_to_experiment_folder(selected_data_directory, os.path.join(experiment_dir, "data"))
                return True, selected_experiment, selected_checkpoint_file

        return False, selected_experiment, selected_checkpoint_file

    def handle_data_split(self, using_checkpoint_data, experiment_dir):
        if using_checkpoint_data:
            return [os.path.join(experiment_dir, "data", d, "pairs.csv") for d in ["train", "test"]]

        dataset_directory       = self.get_data_directory()
        csv_path                = os.path.join(dataset_directory, "pairs.csv")
        data_loader             = DataLoader(dataset_directory, csv_path)
        train_pairs, test_pairs = data_loader.split_data(
            dataset_directory,
            os.path.join(experiment_dir, "data", "train"),
            os.path.join(experiment_dir, "data", "test"),
            csv_path,
            train_ratio=self.config["hyperparameters"]["TRAIN_RATIO"]
        )
        paths = [os.path.join(experiment_dir, "data", d, "pairs.csv") for d in ["train", "test"]]
        for data, path in zip([train_pairs, test_pairs], paths):
            data.to_csv(path, index=False, header=False)

        return paths
    
    def get_data_directory(self):
        if self.check_data_exists(self.base_dir):
            if query_yes_no(f"Would you like to use the default path ({self.base_dir}) for the data set directory? "):
                return self.base_dir
        return input("Please enter the path to the data set directory: ")
    
    @staticmethod
    def check_data_exists(directory):
        return os.path.exists(directory) and any(file.endswith(".csv") for file in os.listdir(directory))

    @staticmethod
    def is_data_already_split(directory):
        return all(os.path.exists(os.path.join(directory, d)) for d in ["train", "test"])

    @staticmethod
    def copy_data_to_experiment_folder(source_dir, dest_dir):
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        shutil.copytree(source_dir, dest_dir)

    @staticmethod
    def create_training_info_file(experiment_dir, **kwargs):
        with open(os.path.join(experiment_dir, "training_info.txt"), "w") as file:
            for key, value in kwargs.items():
                file.write(f"{key.replace('_', ' ').title()}: {value}\n")
     
    def train_model(self, train_csv_path, test_csv_path, experiment_dir, checkpoint_result):
        train_data_loader = DataLoader(train_csv_path, train_csv_path)
        train_dataset     = Dataset(train_data_loader, self.config["hyperparameters"]["BUFFER_SIZE"], self.config["hyperparameters"]["BATCH_SIZE"]).create_dataset()
        test_data_loader  = DataLoader(test_csv_path, test_csv_path)
        test_dataset      = Dataset(test_data_loader, self.config["hyperparameters"]["BUFFER_SIZE"], self.config["hyperparameters"]["BATCH_SIZE"]).create_dataset()

        generator     = Generator()
        discriminator = Discriminator()
        generator.build_model()
        discriminator.build_model()

        with open(os.path.join(experiment_dir, "hyperparameters.yaml"), "w") as file:
            yaml.dump(self.config, file)

        log_dir        = os.path.join(experiment_dir, "logs", "fit")
        checkpoint_dir = os.path.join(experiment_dir, "training_checkpoints")

        summary_writer    = tf.summary.create_file_writer(log_dir)
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        trainer           = Trainer(generator, discriminator, summary_writer, checkpoint_prefix)

        if checkpoint_result:
            selected_experiment, selected_checkpoint_file = checkpoint_result
            checkpoint_path = os.path.join(
                "./experiments",
                selected_experiment,
                "training_checkpoints",
                selected_checkpoint_file.replace('.index', '')
            )
            trainer.checkpoint.restore(checkpoint_path)

        start_time = time.time()
        trainer.fit(train_dataset, test_dataset, steps=self.config["hyperparameters"]["STEPS"], experiment_dir=experiment_dir)
        end_time = time.time()

        total_time       = end_time - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print("Training finished!")


    def orchestrate_training(self):
        try:
            available_checkpoints = self.get_available_checkpoints()
            if available_checkpoints and query_yes_no("Would you like to resume training from a previous checkpoint?"):
                checkpoint_result = self.prompt_for_checkpoint(available_checkpoints)
            else:
                checkpoint_result = None
            experiment_dir        = self.create_experiment_directory()

            using_checkpoint_data, selected_experiment, selected_checkpoint_file = self.handle_checkpoint_data(checkpoint_result, experiment_dir)
            train_csv_path, test_csv_path = self.handle_data_split(using_checkpoint_data, experiment_dir)


            self.create_training_info_file(
                experiment_dir,
                is_continuation     = bool(checkpoint_result),
                previous_experiment = selected_experiment,
                checkpoint_file     = selected_checkpoint_file,
                use_existing_data   = using_checkpoint_data,
                existing_data_dir   = os.path.join(self.EXPERIMENTS_DIR, selected_experiment, "data") if using_checkpoint_data else None
            )

            self.train_model(train_csv_path, test_csv_path, experiment_dir, checkpoint_result)
        except Exception as e:
            print("-" * 50)
            print(f"An error occurred: {e}")
            print("Traceback:")
            traceback.print_exc()



class EvaluatorManager(ModelManager):
    def __init__(self, trainer: Trainer, *args, **kwargs):  
        super().__init__(*args, **kwargs)
        self.trainer = trainer 

    def evaluate_model(self, test_csv_path, checkpoint_result):
        test_data_loader = DataLoader(test_csv_path, test_csv_path)
        test_dataset     = Dataset(test_data_loader, self.config["hyperparameters"]["BUFFER_SIZE"], self.config["hyperparameters"]["BATCH_SIZE"]).create_dataset()

        generator     = Generator()
        discriminator = Discriminator()
        generator.build_model()
        discriminator.build_model()

        selected_experiment, selected_checkpoint_file = checkpoint_result
        checkpoint_path = os.path.join(
            "./experiments",
            selected_experiment,
            "training_checkpoints",
            selected_checkpoint_file.replace('.index', '')
        )
        checkpoint = tf.train.Checkpoint(
            generator_optimizer     = self.trainer.generator_optimizer,
            discriminator_optimizer = self.trainer.discriminator_optimizer,
            generator               = self.trainer.generator.model,
            discriminator           = self.trainer.discriminator.model
        )
        
        status = checkpoint.restore(checkpoint_path)
        status.expect_partial()

        mse_losses = []
        for idx, (file_name, input_image, target) in enumerate(test_dataset):
            prediction = generator.model(input_image, training=False)
            mse_loss   = tf.keras.losses.MeanSquaredError()(target, prediction)
            mse_losses.append(mse_loss.numpy())
            print(f"MSE for test file {file_name.numpy()[0].decode('utf-8')}: {mse_loss.numpy()}")

        avg_mse = sum(mse_losses) / len(mse_losses)
        print(f"Average MSE on test data: {avg_mse}")


    def orchestrate_evaluation(self):
        try:
            available_checkpoints = self.get_available_checkpoints()
            if available_checkpoints:
                print("Please select the checkpoint you want to evaluate from: ")
                checkpoint_result = self.prompt_for_checkpoint(available_checkpoints)
            else: 
                print("No checkpoints found. Please train a model first.")
                return

            using_existing_data = False
            if checkpoint_result:
                selected_experiment, _ = checkpoint_result
                using_existing_data = self.check_for_data(selected_experiment)

            if using_existing_data:
                test_csv_path = os.path.join("./experiments", selected_experiment, "data", "test", "pairs.csv")
            else:
                test_csv_path = os.path.join(self.base_dir, "test", "pairs.csv")
                if not os.path.exists(test_csv_path):
                    print("No test data found in the default directory.")
                    test_csv_path = input("Please enter the path to the test data directory: ")

            self.evaluate_model(test_csv_path, checkpoint_result)
        except Exception as e:
            print("-" * 50)
            print(f"An error occurred: {e}")
            print("Traceback:")
            traceback.print_exc()



def main():
    action = input("Do you want to train or evaluate? (t/e): ").strip().lower()

    while action not in ["t", "e"]:
        print("Invalid input. Please enter t or e.")
        action = input("Do you want to train or evaluate? (t/e): ").strip().lower()

    if action == "t":
        training_manager = TrainingManager()
        training_manager.orchestrate_training()
    elif action == "e":
        generator     = Generator()
        discriminator = Discriminator()
        generator.build_model()
        discriminator.build_model()
        
        experiment_dir    = os.path.join(ModelManager.EXPERIMENTS_DIR, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        log_dir           = os.path.join(experiment_dir, "logs", "fit")
        checkpoint_dir    = os.path.join(experiment_dir, "training_checkpoints")
        summary_writer    = tf.summary.create_file_writer(log_dir)
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        
        trainer = Trainer(generator, discriminator, summary_writer, checkpoint_prefix)

        evaluator_manager = EvaluatorManager(trainer)
        evaluator_manager.orchestrate_evaluation()



if __name__ == "__main__":
    main()