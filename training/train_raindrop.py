# train_raindrop.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from experts.unet_raindrop import UNet
from dataset.paired_dataset import PairedImageDataset
from torchvision.utils import save_image
from tqdm import tqdm

# Device setup for Mac M1
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_raindrop_model():
    model = UNet().to(device)

    train_dataset = PairedImageDataset('dataset/data/raindrop/train')
    val_dataset = PairedImageDataset('dataset/data/raindrop/val')

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    os.makedirs('checkpoints/raindrop', exist_ok=True)

    for epoch in range(100):
        model.train()
        running_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/100")

        for noisy, clean in loop:
            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()

            # MPS needs this to avoid out-of-sync bugs
            if device.type == "mps":
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Save model every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'checkpoints/raindrop/epoch_{epoch+1}.pth')

        # Save one val image for inspection
        model.eval()
        with torch.no_grad():
            for noisy, _ in val_loader:
                noisy = noisy.to(device)
                output = model(noisy)
                save_image(output, f"checkpoints/raindrop/sample_epoch_{epoch+1}.png")
                break  # Only one image per epoch

if __name__ == "__main__":
    train_raindrop_model()