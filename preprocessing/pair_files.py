import os
import re
import csv

def extract_ar_and_ti(filename):
    """Extract the active region and time interval from the filename."""
    ar_match = re.search(r"AR(\d+)", filename)
    # Capture the number with its sign. If no sign is present, the number is treated as positive.
    ti_match = re.search(r"TI([+-]?\d+)", filename)

    if ar_match and ti_match:
        return ar_match.group(1), int(ti_match.group(1))
    return None, None


def get_file_pairs(directory):
    """Get file pairs based on the rules provided."""
    files = os.listdir(directory)
    ar_ti_map = {}

    for file in files:
        ar, ti = extract_ar_and_ti(file)
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


def write_to_csv(directory, save_directory, output_file="pairs.csv"):
    """Write the file pairs to a CSV file."""
    pairs = get_file_pairs(directory)
    output_path = os.path.join(save_directory, output_file)

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for pair in pairs:
            writer.writerow(pair)


if __name__ == "__main__":
    directory = input("Enter the directory path: ")
    save_directory = input("Enter the directory to save the CSV: ")
    output_file = "pairs.csv"

    write_to_csv(directory, save_directory, output_file)
    print(f"File pairs written to {os.path.join(save_directory, output_file)}")
