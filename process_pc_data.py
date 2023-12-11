import os
import shutil

# Directories
source_directory = 'cars'
destination_directory = 'kaggle_data/plane_and_car_train'

# Create destination directory if it doesn't exist
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

counter = 1

# Loop through each subdirectory in the source directory
for subdir in os.listdir(source_directory):
    subdir_path = os.path.join(source_directory, subdir)

    # Skip if it's not a directory
    if not os.path.isdir(subdir_path):
        continue

    # Loop through each file in the subdirectory
    for file in os.listdir(subdir_path):
        # Construct full file path
        file_path = os.path.join(subdir_path, file)

        # Check if it's a file and has a .jpg extension
        if os.path.isfile(file_path) and file.lower().endswith('.jpg'):
            # Construct new file name and path
            new_file_name = f'car{counter}.jpg'
            new_file_path = os.path.join(destination_directory, new_file_name)

            # Copy the file to the new location with the new name
            shutil.copy(file_path, new_file_path)

            counter += 1

