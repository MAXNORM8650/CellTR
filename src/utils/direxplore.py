import os

def explore_directory(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        print(f'Current Path: {dirpath}')
        if dirnames:
            print(f'Directories: {dirnames}')
        if filenames:
            print(f'Files: {filenames}')
        print('---')

# Specify the root directory
root_directory = '/home/komal.kumar/Documents/Cell/datasets/data/CTC/Training/Fluo-N2DL-HeLa'

# Explore the directory
explore_directory(root_directory)
