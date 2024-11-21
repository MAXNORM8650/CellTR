import matplotlib
# matplotlib.use('TkAgg')  # Or 'Qt5Agg', 'Qt4Agg', depending on your system
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os


def read_image(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)
    return image_array

# # Example usage
# image_path = '/home/komal.kumar/Documents/Cell/datasets/data/CTC/Training/Fluo-N2DL-HeLa/01/t000.tif'
# image_array = read_image(image_path)
# print(f'Image shape: {image_array.shape}')

def read_segmentation_mask(mask_path):
    mask = Image.open(mask_path)
    mask_array = np.array(mask)
    return mask_array
# # Example usage
# mask_path = "/home/komal.kumar/Documents/Cell/datasets/data/CTC/Training/Fluo-N2DL-HeLa/01_ST/SEG/man_seg000.tif"
# mask_array = read_segmentation_mask(mask_path)
# print(f'Masks shape: {mask_array.shape}')

def read_tracking_annotations(tracking_file_path):
    tracking_data = {}
    with open(tracking_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Assuming the format: cell_id frame x y [additional info]
            parts = line.strip().split()
            cell_id = int(parts[0])
            frame = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            tracking_data.setdefault(cell_id, []).append((frame, x, y))
    return tracking_data

# Example usage
# tracking_file = "/home/komal.kumar/Documents/Cell/datasets/data/CTC/Training/Fluo-N2DL-HeLa/01_GT/TRA/man_track.txt"
# tracking_data = read_tracking_annotations(tracking_file)
# print(f'Number of cells tracked: {len(tracking_data)}')
# print(tracking_data)



def parse_man_track_file(man_track_file_path):
    import pandas as pd

    # Read the file into a pandas DataFrame
    df = pd.read_csv(man_track_file_path, sep='\s+', comment='#', header=None)
    df.columns = ['label', 't1', 't2', 'parent']

    # Convert DataFrame to a dictionary for easy access
    tracking_data = {}
    for _, row in df.iterrows():
        label = int(row['label'])
        t1 = int(row['t1'])
        t2 = int(row['t2'])
        parent = int(row['parent'])
        tracking_data[label] = {'t1': t1, 't2': t2, 'parent': parent}

    return tracking_data

def extract_cells_from_sequence(sequence_dir, annotation_dir, tracking_data, crop_size=(64, 64)):
    cells_data = []
    image_files = sorted([f for f in os.listdir(sequence_dir) if f.endswith('.tif')])
    annotation_files = sorted([f for f in os.listdir(annotation_dir) if f.endswith('.tif')])

    for img_file, ann_file in zip(image_files, annotation_files):
        image_path = os.path.join(sequence_dir, img_file)
        annotation_path = os.path.join(annotation_dir, ann_file)

        # Extract frame number from image file name (assuming format 'tXXX.tif')
        frame_number_str = os.path.splitext(img_file)[0].lstrip('t')
        frame_number = int(frame_number_str)

        image = Image.open(image_path)
        mask = Image.open(annotation_path)

        # Convert images to arrays
        image_array = np.array(image)
        mask_array = np.array(mask)

        # Extract individual cells based on unique labels in the mask
        cell_labels = np.unique(mask_array)
        cell_labels = cell_labels[cell_labels != 0]  # Exclude background

        for label in cell_labels:
            # Check if the cell is present in the current frame based on tracking data
            cell_info = tracking_data.get(label)
            if cell_info is None:
                continue  # Skip if cell info is not available

            if not (cell_info['t1'] <= frame_number <= cell_info['t2']):
                continue  # Skip if the cell is not present in this frame

            cell_mask = (mask_array == label).astype(np.uint8)
            positions = np.argwhere(cell_mask)
            if positions.size == 0:
                continue  # Skip if the mask is empty

            y_min, x_min = positions.min(axis=0)
            y_max, x_max = positions.max(axis=0) + 1  # +1 to include the max index

            # Crop the image
            cell_image = image_array[y_min:y_max, x_min:x_max]

            # Handle image modes and convert to 'L' or 'RGB'
            if cell_image.ndim == 2:
                # Grayscale image
                cell_image_pil = Image.fromarray(cell_image)
            elif cell_image.ndim == 3:
                # Multichannel image
                cell_image_pil = Image.fromarray(cell_image)
            else:
                print(f'Unexpected image dimensions: {cell_image.shape}')
                continue

            # Convert image mode if necessary
            if cell_image_pil.mode not in ['L', 'RGB']:
                if cell_image_pil.mode in ['I;16', 'I;16B']:
                    # Normalize the 16-bit image to 8-bit
                    np_image = np.array(cell_image_pil, dtype=np.float32)
                    np_image = (np_image - np_image.min()) / (np_image.max() - np_image.min()) * 255.0
                    np_image = np_image.astype(np.uint8)
                    cell_image_pil = Image.fromarray(np_image, mode='L')
                else:
                    # Convert other modes to 'L'
                    cell_image_pil = cell_image_pil.convert('L')

            # Resize crop to fixed size
            cell_image_resized = cell_image_pil.resize(crop_size, resample=Image.LANCZOS)

            # Store the cell data
            cells_data.append({
                'cell_id': label,
                'parent_id': cell_info['parent'],
                't1': cell_info['t1'],
                't2': cell_info['t2'],
                'frame': frame_number,
                'image': np.array(cell_image_resized),
                'mask': cell_mask[y_min:y_max, x_min:x_max]
            })

    return cells_data

# Example usage
sequence_dir = '/home/komal.kumar/Documents/Cell/datasets/data/CTC/Training/Fluo-N2DL-HeLa/01/'
annotation_dir = '/home/komal.kumar/Documents/Cell/datasets/data/CTC/Training/Fluo-N2DL-HeLa/01_ST/SEG'
man_track_file = "/home/komal.kumar/Documents/Cell/datasets/data/CTC/Training/Fluo-N2DL-HeLa/01_GT/TRA/man_track.txt"

# Parse tracking data
tracking_data = parse_man_track_file(man_track_file)

# Extract cells with tracking information
cells_data = extract_cells_from_sequence(sequence_dir, annotation_dir, tracking_data)

print(f'Extracted {len(cells_data)} cell instances')

def visualize_cells(cells_data, num_samples=5, save_dir='cell_images'):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(num_samples, len(cells_data))):
        cell = cells_data[i]
        plt.figure()
        plt.imshow(cell['image'], cmap='gray')
        plt.title(f'Cell ID: {cell["cell_id"]} Frame: {cell["frame"]}')
        plt.axis('off')
        filename = f'cell_{cell["cell_id"]}_frame_{cell["frame"]}.png'
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        print(f'Saved cell visualization to {filepath}')
# Assuming 'cells_data' is already populated
visualize_cells(cells_data, num_samples=10)

# Visualize some cells
visualize_cells(cells_data)
