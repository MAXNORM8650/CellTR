import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.utils import save_image
from torchvision.transforms import functional as TF
from PIL import Image, ImageSequence
import albumentations as A
from albumentations.pytorch import ToTensorV2
# -------------------------------
# Utility Functions
# -------------------------------
def load_single_frame_tiff(path, frame_number=0, mode='L'):
    """
    Loads a single frame from a multi-page TIFF file.
    Args:
        path (str): Path to the TIFF file.
        frame_number (int): Index of the frame to load.
        mode (str): Mode to convert the image ('RGB' for frames, 'L' for masks).
    Returns:
        PIL.Image: Converted image.
    """
    with Image.open(path) as img:
        try:
            frame = ImageSequence.Iterator(img).__getitem__(frame_number)
            return frame.convert(mode)
        except IndexError:
            # If frame_number is out of range, return the first frame
            return img.convert(mode)

# -------------------------------
# Dataset Class
# -------------------------------
class CellSequenceDataset(Dataset):
    def __init__(self, frames_dir, masks_dir, sequence_length=5, transform=None, augmentation=True, image_size=(512, 512)):
        """
        Args:
            frames_dir (str): Directory with all the frame .tiff images.
            masks_dir (str): Directory with all the mask .tiff images.
            sequence_length (int): Number of frames/masks in the input sequence.
            transform (callable, optional): Optional transform to be applied on a sample.
            augmentation (bool): Whether to apply data augmentation.
            image_size (tuple): Desired output image size (height, width).
        """
        self.frames_dir = frames_dir
        self.masks_dir = masks_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.augmentation = augmentation
        self.image_size = image_size

        # List all .tiff and .tif files and sort them
        self.frame_files = sorted([
            f for f in os.listdir(frames_dir) if f.lower().endswith('.tiff') or f.lower().endswith('.tif')
        ])
        self.mask_files = sorted([
            f for f in os.listdir(masks_dir) if f.lower().endswith('.tiff') or f.lower().endswith('.tif')
        ])

        # Augmentation pipeline
        self.augmentation_pipeline = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.7),
            A.ElasticTransform(p=0.5),
            A.Resize(height=self.image_size[0], width=self.image_size[1]),  # Ensure consistent size
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2()
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

            # Load frames and masks with specified modes
            frame = load_single_frame_tiff(frame_path, mode='L')  # 1 channel
            mask = load_single_frame_tiff(mask_path, mode='L')    # 1 channel

            # Apply synchronized transformations
            if self.augmentation:
                frame, mask = self._synchronized_transform(frame, mask)
            elif self.transform:
                frame = self.transform(frame)
                mask = self.transform(mask)

            frames.append(frame)
            masks.append(mask)

        # The target is the mask of the next frame
        target_mask_path = os.path.join(self.masks_dir, self.mask_files[idx + self.sequence_length])
        target_mask = load_single_frame_tiff(target_mask_path, mode='L')  # Ensure single channel
        if self.augmentation:
            # Apply the same resizing and normalization to the target mask
            target_transform = A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2()
            ])
            augmented = target_transform(image=np.array(target_mask))
            target_mask = augmented['image']
        elif self.transform:
            target_mask = self.transform(target_mask)

        frames = torch.stack(frames)        # Shape: [sequence_length, C, H, W]
        masks = torch.stack(masks)          # Shape: [sequence_length, C, H, W]

        return {'frames': frames, 'masks': masks, 'target_mask': target_mask}


    def _synchronized_transform(self, frame, mask):
        augmented = self.augmentation_pipeline(image=np.array(frame), mask=np.array(mask))
        frame = augmented['image']
        mask = augmented['mask']
        return frame, mask
