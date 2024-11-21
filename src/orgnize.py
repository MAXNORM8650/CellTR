import os
import shutil

# Define the source and target directories
target_dir = "/home/komal.kumar/Documents/Cell/src/GIF"
source_dir = "/home/komal.kumar/Documents/Cell/src"

# Ensure the target directory exists, create if not
os.makedirs(target_dir, exist_ok=True)

# Iterate through all files in the source directory
for file_name in os.listdir(source_dir):
    # Check if the file is a .gif
    if file_name.endswith(".gif"):
        # Get the full path of the source file
        source_file = os.path.join(source_dir, file_name)
        
        # Define the target file path
        target_file = os.path.join(target_dir, file_name)
        
        # Move the file to the target directory
        shutil.move(source_file, target_file)

print(f"All GIF files moved to {target_dir}")
