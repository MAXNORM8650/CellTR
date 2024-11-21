import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def extract_cells(I_t, annotations_t, crop_size=(64, 64)):
    """
    Extract cell crops from image I_t based on annotations_t.

    Args:
        I_t (torch.Tensor): The image tensor at time t with shape (C, H, W).
        annotations_t (list): List of annotations for time t.
            Each annotation is a dictionary with keys:
                - 'bbox': [x_min, y_min, x_max, y_max]
                - 'cell_id': int
        crop_size (tuple): Desired size (H, W) to resize the cell crops.

    Returns:
        cells_t (list): List of dictionaries containing:
            - 'cell_id': int
            - 'crop': torch.Tensor of the cell crop resized to crop_size.
    """
    cells_t = []
    # Ensure the image tensor is on CPU and detached
    I_t_cpu = I_t.detach().cpu()

    for ann in annotations_t:
        # Extract bounding box coordinates
        bbox = ann.get('bbox')
        if bbox is None:
            continue  # Skip if no bounding box is provided

        x_min, y_min, x_max, y_max = bbox

        # Ensure coordinates are integers and within image bounds
        x_min = int(max(0, x_min))
        y_min = int(max(0, y_min))
        x_max = int(min(I_t_cpu.shape[2], x_max))
        y_max = int(min(I_t_cpu.shape[1], y_max))

        # Crop the image
        crop = I_t_cpu[:, y_min:y_max, x_min:x_max]  # Shape: (C, crop_H, crop_W)

        # Resize crop to fixed size
        resize_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(crop_size),
            transforms.ToTensor()
        ])
        crop_resized = resize_transform(crop)

        # Append to the list
        cells_t.append({'cell_id': ann['cell_id'], 'crop': crop_resized})

    return cells_t
