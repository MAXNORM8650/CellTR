import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision.utils import save_image

import numpy as np
import os
from PIL import Image
from tqdm import tqdm

import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class CellSequenceDataset(Dataset):
    def __init__(self, frames_dir, masks_dir, sequence_length=5, transform=None):
        """
        Args:
            frames_dir (str): Directory with all the frame .tiff images.
            masks_dir (str): Directory with all the mask .tiff images.
            sequence_length (int): Number of frames/masks in the input sequence.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.frames_dir = frames_dir
        self.masks_dir = masks_dir
        self.sequence_length = sequence_length
        self.transform = transform

        # List all .tiff files and sort them
        self.frame_files = sorted([
            f for f in os.listdir(frames_dir) if f.lower().endswith('.tiff') or f.lower().endswith('.tif')
        ])
        self.mask_files = sorted([
            f for f in os.listdir(masks_dir) if f.lower().endswith('.tiff') or f.lower().endswith('.tif')
        ])
        
        assert len(self.frame_files) == len(self.mask_files), "Number of frames and masks do not match."

    def __len__(self):
        return len(self.frame_files) - self.sequence_length

    def __getitem__(self, idx):
        frames = []
        masks = []
        for i in range(self.sequence_length):
            frame_path = os.path.join(self.frames_dir, self.frame_files[idx + i])
            mask_path = os.path.join(self.masks_dir, self.mask_files[idx + i])

            # Open .tiff images
            frame = Image.open(frame_path).convert('RGB')  # Assuming frames are RGB
            mask = Image.open(mask_path).convert('L')      # Assuming masks are grayscale

            if self.transform:
                frame = self.transform(frame)
                mask = self.transform(mask)

            frames.append(frame)
            masks.append(mask)

        # The target is the mask of the next frame
        target_mask_path = os.path.join(self.masks_dir, self.mask_files[idx + self.sequence_length])
        target_mask = Image.open(target_mask_path).convert('L')
        if self.transform:
            target_mask = self.transform(target_mask)

        frames = torch.stack(frames)        # Shape: [sequence_length, C, H, W]
        masks = torch.stack(masks)          # Shape: [sequence_length, 1, H, W]

        return {'frames': frames, 'masks': masks, 'target_mask': target_mask}


# Define transformations for frames and masks
transform = T.Compose([
    T.Resize((128, 128)),  # Resize to desired dimensions
    T.ToTensor(),
    # For frames (RGB), use mean and std for 3 channels
    # For masks (grayscale), use mean and std for 1 channel
    # We'll handle this dynamically in the dataset if needed
])

# Initialize the dataset with your directories
sequence_dir = '/home/komal.kumar/Documents/Cell/datasets/data/CTC/Training/Fluo-N2DL-HeLa/01/'
annotation_dir = '/home/komal.kumar/Documents/Cell/datasets/data/CTC/Training/Fluo-N2DL-HeLa/01_ST/SEG'

dataset = CellSequenceDataset(
    frames_dir=sequence_dir,
    masks_dir=annotation_dir,
    sequence_length=5,          # Adjust based on your sequence requirements
    transform=transform
)

# Create the DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=16,               # Adjust based on your GPU memory
    shuffle=True,
    num_workers=4,               # Adjust based on your CPU cores
    pin_memory=True              # Improves data transfer speed to GPU
)

# Initialize diffusion configuration
config = DiffusionConfig(timesteps=1000, beta_start=1e-4, beta_end=0.02)

# Number of conditioning channels
sequence_length = 5
condition_channels = (3 * sequence_length) + (1 * sequence_length)  # 5 frames * 3 + 5 masks * 1 = 20

# Initialize the diffusion model
model = DiffusionModel(config, condition_channels=condition_channels)
