from PIL import Image
import os

def load_images(image_folder):
    images = []
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith('.tif'):
            img_path = os.path.join(image_folder, filename)
            images.append(Image.open(img_path))
    return images

def load_masks(mask_folder):
    masks = []
    for filename in sorted(os.listdir(mask_folder)):
        if filename.endswith('.tif'):
            mask_path = os.path.join(mask_folder, filename)
            masks.append(Image.open(mask_path))
    return masks

# Load image and mask sequences
image_folder = "/home/komal.kumar/Documents/Cell/datasets/data/CTC/Training/Fluo-N2DL-HeLa/01"
mask_folder = "/home/komal.kumar/Documents/Cell/datasets/data/CTC/Training/Fluo-N2DL-HeLa/01_ST/SEG"

images = load_images(image_folder)
masks = load_masks(mask_folder)
# breakpoint()

def load_tracking_info(tracking_file):
    tracking_data = []
    with open(tracking_file, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue  # Skip comments
            parts = line.strip().split()
            frame = int(parts[0])
            cell_id = int(parts[1])
            x_pos = float(parts[2])
            y_pos = float(parts[3])
            tracking_data.append((frame, cell_id, x_pos, y_pos))
    return tracking_data

tracking_file = "/home/komal.kumar/Documents/Cell/datasets/data/CTC/Training/Fluo-N2DL-HeLa/01_GT/TRA/man_track.txt"
tracking_info = load_tracking_info(tracking_file)
breakpoint()
def compute_actions(tracking_info):
    actions = []
    for i in range(1, len(tracking_info)):
        prev_x, prev_y = tracking_info[i-1][2], tracking_info[i-1][3]
        curr_x, curr_y = tracking_info[i][2], tracking_info[i][3]
        # Compute movement vector (action)
        dx = curr_x - prev_x
        dy = curr_y - prev_y
        actions.append((dx, dy))
    return actions

actions = compute_actions(tracking_info)
