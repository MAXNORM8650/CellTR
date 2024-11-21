import torch
from torch.utils.data import Dataset, DataLoader

class CellTrackingDataset(Dataset):
    def __init__(self, image_sequences, annotations):
        self.image_sequences = image_sequences
        self.annotations = annotations

    def __len__(self):
        return len(self.image_sequences)

    def __getitem__(self, idx):
        images = self.image_sequences[idx]
        targets = self.annotations[idx]
        return images, targets

# Create dataset and dataloader
dataset = CellTrackingDataset(image_sequences, annotations)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
