import os
import shutil

class FileManager:
    """
    The FileManager class provides static methods for file and directory management operations.
    """
    
    @staticmethod
    def check_data_exists(directory):
        """
        Checks if the specified directory exists and contains any CSV files.

        Args:
            directory (str): The path to the directory to be checked.

        Returns:
            bool: True if the directory exists and contains at least one CSV file, otherwise False.
        """
        return os.path.exists(directory) and any(file.endswith(".csv") for file in os.listdir(directory))

    @staticmethod
    def copy_data_to_folder(source_dir, dest_dir):
        """
        Copies the contents of the source directory to the destination directory. If the destination directory exists, it is removed before copying.

        Args:
            source_dir (str): The path to the source directory.
            dest_dir (str): The path to the destination directory.
        """
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        shutil.copytree(source_dir, dest_dir)
