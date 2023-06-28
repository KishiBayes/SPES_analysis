import os
import pyedflib
from Peak_Detection import summed_peaks_simple

def crawl_folder_for_files(folder_path, destination_folder):
    # Specify the folder path to crawl
    folder_path = os.path.abspath(folder_path)

    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Create an empty list to store the paths of eligible files
    eligible_files = []

    # Walk through the folder and its subdirectories
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if the file is an EDF and starts with "SPES"
            if file.lower().endswith('.edf') and file.startswith('SPES'):
                file_path = os.path.join(root, file)

                # Load the EDF file and apply summed_peaks_simple function
                try:
                    edf_data = pyedflib.EdfReader(file_path)
                    data = edf_data.readSignal(0)  # Assuming you want to process the first signal
                    result = summed_peaks_simple(data)  # Apply your function here

                    # Check if the result meets the criteria
                    if sum(result) >= 20:
                        eligible_files.append(file_path)

                        # Copy the file to the destination folder
                        destination_path = os.path.join(destination_folder, file)
                        shutil.copy2(file_path, destination_path)
                        print(f"Copied {file} to {destination_path}")

                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    # Return the list of eligible file paths
    return eligible_files

# Example usage
folder_path = '/path/to/your/folder'
destination_folder = '/path/to/destination/folder'
eligible_files = crawl_folder_for_files(folder_path, destination_folder)

# Print the eligible file paths
for file_path in eligible_files:
    print(file_path)