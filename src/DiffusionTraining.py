import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

def train_diffusion_model(model, dataloader, config, epochs, device):
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    mse = nn.MSELoss()

    model.to(device)
    config.beta = config.beta.to(device)
    config.alpha = config.alpha.to(device)
    config.alpha_bar = config.alpha_bar.to(device)

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0
        for batch in pbar:
            frames = batch['frames'].to(device)         # [B, S, 3, H, W]
            masks = batch['masks'].to(device)           # [B, S, 1, H, W]
            target_mask = batch['target_mask'].to(device)  # [B, 1, H, W]

            B, S, C, H, W = frames.shape

            # Prepare conditioning input by flattening the sequence
            # Concatenate frames and masks along the channel dimension
            condition = torch.cat([
                frames.view(B, -1, H, W),   # [B, S*3, H, W]
                masks.view(B, -1, H, W)     # [B, S*1, H, W]
            ], dim=1)  # Total channels: 3S + 1S = 4S (e.g., 20 channels)

            # Sample random timesteps
            t = torch.randint(0, config.timesteps, (B,), device=device).long()

            # Sample noise
            noise = torch.randn_like(target_mask)
            alpha_bar_t = config.alpha_bar[t].view(B, 1, 1, 1)
            noisy_mask = torch.sqrt(alpha_bar_t) * target_mask + torch.sqrt(1 - alpha_bar_t) * noise

            # Predict the noise
            predicted_noise = model(noisy_mask, condition, t)

            loss = mse(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'Loss': loss.item()})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

        # Optionally, save model checkpoints
        torch.save(model.state_dict(), f'diffusion_model_epoch_{epoch+1}.pth')
def main():
    import torchvision.transforms as T

    # Configuration
    sequence_dir = '/home/komal.kumar/Documents/Cell/datasets/data/CTC/Training/Fluo-N2DL-HeLa/01/'
    annotation_dir = '/home/komal.kumar/Documents/Cell/datasets/data/CTC/Training/Fluo-N2DL-HeLa/01_ST/SEG'
    sequence_length = 5
    batch_size = 16
    epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define transformations
    transform = T.Compose([
        T.Resize((128, 128)),        # Resize to desired dimensions
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])  # Normalize masks
    ])

    # Initialize dataset and dataloader
    dataset = CellSequenceDataset(
        frames_dir=sequence_dir,
        masks_dir=annotation_dir,
        sequence_length=sequence_length,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Initialize diffusion configuration
    config = DiffusionConfig(timesteps=1000, beta_start=1e-4, beta_end=0.02)

    # Calculate conditioning channels
    condition_channels = (3 * sequence_length) + (1 * sequence_length)  # e.g., 5*3 + 5*1 = 20

    # Initialize model
    model = DiffusionModel(config, condition_channels=condition_channels)

    # Train the model
    train_diffusion_model(model, dataloader, config, epochs, device)

    # Save final model
    torch.save(model.state_dict(), 'diffusion_model_final.pth')

if __name__ == "__main__":
    main()
