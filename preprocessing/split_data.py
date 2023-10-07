import os
import shutil
import pathlib
import pandas as pd

class DataSplitter:
    """
    A class to split a dataset into training and testing datasets based on a provided ratio.

    Attributes:
    - dataset_path (str): Path to the dataset directory.
    - train_path (str): Path to the training dataset directory.
    - test_path (str): Path to the testing dataset directory.
    - csv_path (str): Path to the CSV file containing pairs of data.
    - train_ratio (float): Ratio of data to be used for training.
    """

    def __init__(self, dataset_path, train_path, test_path, csv_path, train_ratio=0.85):
        self.dataset_path = pathlib.Path(dataset_path)
        self.train_path = pathlib.Path(train_path)
        self.test_path = pathlib.Path(test_path)
        self.csv_path = csv_path
        self.train_ratio = train_ratio

    @staticmethod
    def _clear_directory(directory):
        """
        Clears the contents of a directory.

        Parameters:
        - directory (pathlib.Path): Path to the directory to be cleared.
        """
        for item in directory.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    def split(self):
        """
        Splits the dataset into training and testing datasets.
        """
        # Ensure train and test directories exist
        self.train_path.mkdir(parents=True, exist_ok=True)
        self.test_path.mkdir(parents=True, exist_ok=True)

        # Clear the contents of train and test directories if they exist and have data
        if any(self.train_path.iterdir()):
            self._clear_directory(self.train_path)
        if any(self.test_path.iterdir()):
            self._clear_directory(self.test_path)

        # Read the CSV file into a DataFrame and shuffle the pairs
        pairs = pd.read_csv(self.csv_path, header=None)
        pairs = pairs.sample(frac=1).reset_index(drop=True)

        # Split the shuffled pairs DataFrame into training and testing datasets
        train_size = int(self.train_ratio * len(pairs))
        train_pairs = pairs.iloc[:train_size]
        test_pairs = pairs.iloc[train_size:]

        # Copy the files based on the split pairs to train and test directories
        for _, row in train_pairs.iterrows():
            shutil.copy(self.dataset_path / row.iloc[0], self.train_path / row.iloc[0])
            shutil.copy(self.dataset_path / row.iloc[1], self.train_path / row.iloc[1])

        for _, row in test_pairs.iterrows():
            shutil.copy(self.dataset_path / row.iloc[0], self.test_path / row.iloc[0])
            shutil.copy(self.dataset_path / row.iloc[1], self.test_path / row.iloc[1])

        # Save the train_pairs and test_pairs as CSV files in the train and test directories
        train_pairs.to_csv(os.path.join(self.train_path, 'pairs.csv'), index=False, header=None)
        test_pairs.to_csv(os.path.join(self.test_path, 'pairs.csv'), index=False, header=None)

    @classmethod
    def from_user_input(cls):
        """
        Creates an instance of the DataSplitter class based on user input.
        """
        # Get user input for the training data percentage
        try:
            train_percentage = float(input("Enter the percentage (0 - 100) of data to be used for training (default is 85, meaning 85% of the data will be training data and 15% testing data): "))
            if not (0 <= train_percentage <= 100):
                raise ValueError
            train_ratio = train_percentage / 100
        except ValueError:
            print("Invalid input. Using default value of 85% for training.")
            train_ratio = 0.85

        # Get user input for the directory path
        dataset_directory = input("Enter the directory path to the data (should contain the original pairs.csv file and data files): ")
        csv_path = os.path.join(dataset_directory, "pairs.csv")
        if not os.path.exists(csv_path):
            print(f"Error: {csv_path} does not exist.")
            return

        # Get user input for where to save the test and train folders
        save_directory = input("Enter the directory path where you'd like to save the test and train folders: ")
        train_path = os.path.join(save_directory, "train")
        test_path = os.path.join(save_directory, "test")

        return cls(dataset_directory, train_path, test_path, csv_path, train_ratio)

if __name__ == "__main__":
    data_splitter = DataSplitter.from_user_input()
    if data_splitter:
        data_splitter.split()
        print("Data split successfully!")
