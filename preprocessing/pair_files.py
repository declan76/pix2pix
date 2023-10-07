import os
import re
import csv

class PairFiles:
    """
    A class to extract file pairs based on specific rules from filenames in a directory.

    Attributes:
        directory (str): Directory containing the files.
        save_directory (str): Directory to save the output CSV file.
        output_file (str): Name of the output CSV file.
    """

    def __init__(self, directory, save_directory, output_file="pairs.csv"):
        """
        Initializes the FilePairExtractor class with directory, save_directory, and output_file.

        Parameters:
            directory (str): Directory containing the files.
            save_directory (str): Directory to save the output CSV file.
            output_file (str, optional): Name of the output CSV file. Defaults to "pairs.csv".
        """
        self.directory = directory
        self.save_directory = save_directory
        self.output_file = output_file

        
    def extract_ar_and_ti(self, filename):
        """
        Extract the active region and time interval from the filename.

        Parameters:
            filename (str): Name of the file.

        Returns:
            tuple: Active region and time interval.
        """
        ar_match = re.search(r"AR(\d+)", filename)
        # Capture the number with its sign. If no sign is present, the number is treated as positive.
        ti_match = re.search(r"TI([+-]?\d+)", filename)

        if ar_match and ti_match:
            return ar_match.group(1), int(ti_match.group(1))
        return None, None


    def get_file_pairs(self):
        """
        Get file pairs based on the rules provided.

        Returns:
            list: List of file pairs.
        """
        files = os.listdir(directory)
        ar_ti_map = {}

        for file in files:
            ar, ti = self.extract_ar_and_ti(file)
            if ar and ti is not None:
                if ar not in ar_ti_map:
                    ar_ti_map[ar] = []
                ar_ti_map[ar].append((ti, file))

        # Sort the time intervals for each active region
        for ar in ar_ti_map:
            ar_ti_map[ar].sort(key=lambda x: x[0])  # Sort by the time interval value

        pairs = []
        for ar, ti_files in ar_ti_map.items():
            for i in range(len(ti_files) - 1):
                # Only pair consecutive time intervals
                if ti_files[i + 1][0] - ti_files[i][0] == 1:
                    pairs.append((ti_files[i][1], ti_files[i + 1][1]))

        return pairs


    def write_to_csv(self):
        """
        Write the file pairs to a CSV file.
        """
        pairs = self.get_file_pairs()
        output_path = os.path.join(self.save_directory, self.output_file)

        with open(output_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for pair in pairs:
                writer.writerow(pair)

    
    def run(self):
        """
        Main method to run the file pair extraction and save to CSV.
        """
        self.write_to_csv()
        print(f"File pairs written to {os.path.join(self.save_directory, self.output_file)}")



if __name__ == "__main__":
    directory = input("Enter the directory path: ")
    save_directory = input("Enter the directory to save the CSV: ")
    output_file = "pairs.csv"

    extractor = PairFiles(directory, save_directory, output_file)
    extractor.run()
