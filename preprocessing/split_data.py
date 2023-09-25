import os
import shutil
import pathlib
import pandas as pd

def split_data(dataset_path, train_path, test_path, csv_path, train_ratio):
        dataset_path_obj = pathlib.Path(dataset_path)
        train_path_obj   = pathlib.Path(train_path)
        test_path_obj    = pathlib.Path(test_path)

        # Function to clear the contents of a directory
        def clear_directory(directory):
            for item in directory.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

        # Ensure train and test directories exist
        train_path_obj.mkdir(parents=True, exist_ok=True)
        test_path_obj.mkdir(parents=True, exist_ok=True)

        # Clear the contents of train and test directories if they exist and have data
        if any(train_path_obj.iterdir()):
            clear_directory(train_path_obj)
        if any(test_path_obj.iterdir()):
            clear_directory(test_path_obj)

        # Read the CSV file into a DataFrame and shuffle the pairs
        pairs = pd.read_csv(csv_path, header=None)  
        pairs = pairs.sample(frac=1).reset_index(drop=True)

        # Split the shuffled pairs DataFrame into training and testing datasets
        train_size  = int(train_ratio * len(pairs))
        train_pairs = pairs.iloc[:train_size]
        test_pairs  = pairs.iloc[train_size:]

        # Copy the files based on the split pairs to train and test directories
        for _, row in train_pairs.iterrows():
            shutil.copy(dataset_path_obj / row.iloc[0], train_path_obj / row.iloc[0])   # Use iloc
            shutil.copy(dataset_path_obj / row.iloc[1], train_path_obj / row.iloc[1])   # Use iloc

        for _, row in test_pairs.iterrows():
            shutil.copy(dataset_path_obj / row.iloc[0], test_path_obj / row.iloc[0])    # Use iloc
            shutil.copy(dataset_path_obj / row.iloc[1], test_path_obj / row.iloc[1])    # Use iloc

        # Save the train_pairs and test_pairs as CSV files in the train and test directories
        train_pairs.to_csv(os.path.join(train_path, 'pairs.csv'), index=False, header=None)
        test_pairs.to_csv(os.path.join(test_path, 'pairs.csv'), index=False, header=None)



def main():
    # Get user input for the training data percentage
    try:
        train_percentage = float(input("Enter the percentage of data to be used for training (default is 85, so 85 will be training data and 15 testing data): "))
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
    train_path     = os.path.join(save_directory, "train")
    test_path      = os.path.join(save_directory, "test")

    # Execute the split_data function
    split_data(dataset_directory, train_path, test_path, csv_path, train_ratio)
    print("Data split successfully!")

if __name__ == "__main__":
    main()

