from torchvision.utils import save_image

def predict_next_mask(model, frames, masks, config, device):
    """
    Args:
        model (DiffusionModel): Trained diffusion model.
        frames (torch.Tensor): Input frames, shape [B, S, 3, H, W]
        masks (torch.Tensor): Input masks, shape [B, S, 1, H, W]
        config (DiffusionConfig): Diffusion configuration.
        device (torch.device): Device to perform computations on.

    Returns:
        torch.Tensor: Predicted masks, shape [B, 1, H, W]
    """
    model.eval()
    with torch.no_grad():
        B, S, C, H, W = frames.shape
        # Concatenate frames and masks along the channel dimension
        condition = torch.cat([
            frames.view(B, -1, H, W),   # [B, S*3, H, W]
            masks.view(B, -1, H, W)     # [B, S*1, H, W]
        ], dim=1).to(device)             # [B, 4S, H, W]

        shape = (B, 1, H, W)
        predicted_mask = model.sample(condition, device, shape)

    return predicted_mask

def example_prediction(model, dataloader, config, device):
    model.to(device)
    model.eval()

    batch = next(iter(dataloader))
    frames = batch['frames'].to(device)          # [B, S, 3, H, W]
    masks = batch['masks'].to(device)            # [B, S, 1, H, W]

    predicted_masks = predict_next_mask(model, frames, masks, config, device)

    # Save or visualize the predicted masks
    for i in range(predicted_masks.size(0)):
        # Denormalize if necessary
        mask = predicted_masks[i] * 0.5 + 0.5  # Assuming normalization was (x - 0.5)/0.5
        save_image(mask, f'predicted_mask_{i}.png')

    print("Predicted masks saved successfully.")
def run_prediction():
    # Load the trained model
    config = DiffusionConfig(timesteps=1000, beta_start=1e-4, beta_end=0.02)
    condition_channels = (3 * 5) + (1 * 5)  # Adjust based on sequence_length
    model = DiffusionModel(config, condition_channels=condition_channels)
    model.load_state_dict(torch.load('diffusion_model_final.pth'))
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Initialize dataset and dataloader (use a separate validation/test set if available)
    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])
    
    dataset = CellSequenceDataset(
        frames_dir=sequence_dir,
        masks_dir=annotation_dir,
        sequence_length=5,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,               # Predict one sequence at a time
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Run prediction
    example_prediction(model, dataloader, config, device)
