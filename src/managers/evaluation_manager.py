import os
import datetime
import traceback
import tensorflow as tf

from astropy.io import fits
from data.dataset import Dataset
from pix2pix.train import Trainer
from utils.pdf_writer import PDFWriter
from data.data_loader import DataLoader
from managers.file_manager import FileManager
from managers.model_manager import ModelManager
from utils.fits_handler import generate_images
from managers.user_input_manager import UserInputManager

class EvaluationManager(ModelManager):
    """
    The EvaluationManager class is responsible for evaluating a trained model using test data.
    It provides methods to evaluate the model, calculate Mean Squared Error (MSE) for test data,
    and save the evaluation results.
    It inherits from the ModelManager class.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the EvaluationManager.
        """
        super().__init__(*args, **kwargs)

    def get_default_evaluation_path(self, checkpoint_path):
        """
        Get the default path for evaluation results based on the checkpoint path.
        
        Args:
            checkpoint_path (str): Path to the model checkpoint.
        
        Returns:
            str: Default path for evaluation results.
        """
        base_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        return os.path.join(base_dir, "evaluation")

    def get_final_save_path(self, checkpoint_path):
        """
        Generate a unique path to save the evaluation results based on the current timestamp and checkpoint name.
        
        Args:
            checkpoint_path (str): Path to the model checkpoint.
        
        Returns:
            tuple: Tuple containing the final save path and the current timestamp.
        """
        timestamp       = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        checkpoint_name = os.path.basename(checkpoint_path)
        folder_name     = f"{timestamp}_{checkpoint_name}"
        return os.path.join(self.get_default_evaluation_path(checkpoint_path), folder_name), timestamp

    def save_evaluation_results(self, final_save_path, mse_values):
        """
        Save the MSE values of the evaluation to a CSV file.
        
        Args:
            final_save_path (str): Path to save the evaluation results.
            mse_values (dict): Dictionary containing file names as keys and their corresponding MSE values as values.
        """
        os.makedirs(final_save_path, exist_ok=True)
        mse_file_path = os.path.join(final_save_path, "MSE.CSV")
        with open(mse_file_path, "w") as file:
            for filename, mse in mse_values.items():
                file.write(f"{filename},{mse}\n")
        print(f"Evaluation results saved to {mse_file_path}")


    def evaluate_model(self, test_csv_path, checkpoint_path, save_images_path):
        """
        Evaluate the trained model using test data and calculate the MSE for each test file.
        
        Args:
            test_csv_path (str): Path to the test data CSV file.
            checkpoint_path (str): Path to the model checkpoint.
            save_images_path (str): Path to save the generated images.
        
        Returns:
            tuple: Tuple containing a dictionary of MSE values and a list of image paths.
        """
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
                image_name_png = f"{taget_name}_predicted.png"
                image_path = os.path.join(save_images_path, image_name_png)
                tf.keras.preprocessing.image.save_img(image_path, prediction[0])

                image_name_fits = f"{taget_name}_predicted.fits" 
                fits_path = os.path.join(save_images_path, image_name_fits)
                prediction_np = prediction[0].numpy()
                fits.writeto(fits_path, prediction_np, overwrite=True)

                generate_images(generator, input, target, taget_name, save_images_path, mode='eval')

        # Create temp folder to store images for PDF report
        os.makedirs("temp", exist_ok=True)
        
        # Identify top 3 and worst MSE files
        sorted_mse_items = sorted(mse_values.items(), key=lambda x: x[1])
        top_3_files      = sorted_mse_items[:3]
        worst_file       = sorted_mse_items[-1]

        # Generate images for these files to be used in the PDF report
        for rank, (file_name, mse) in enumerate(top_3_files + [worst_file]):
            _, input, target = next(filter(lambda x: x[0].numpy()[0].decode('utf-8') == file_name, test_dataset))
            image_path       = generate_images(generator, input, target, file_name, "temp", mode='eval')
            images.append((rank, image_path))

        avg_mse = sum(mse_values.values()) / len(mse_values)
        print(f"Average MSE on test data: {avg_mse}")
        return mse_values, images  


    def orchestrate_evaluation(self):
        """
        Orchestrates the entire evaluation process, including prompting the user for necessary inputs,
        evaluating the model, and saving the results.
        """
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

            PDFWriter.generate_pdf_report(checkpoint_path, timestamp, mse_values, final_save_path, images)

        except Exception as e:
            print("-" * 50)
            print(f"An error occurred: {e}")
            print("Traceback:")
            traceback.print_exc()