# test_ct.py
import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from experts.unet_raindrop import UNet
from dataset.paired_dataset import PairedImageDataset

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Load modelz
model = UNet(out_channels=3).to(device) # change out channel to 3 for RGB 
model.load_state_dict(torch.load("checkpoints/raindrop/epoch_100.pth", map_location=device))
model.eval()

# Dataset
test_dataset = PairedImageDataset('dataset/data/raindrop/test')  # , as_gray=False# change this to False if you want RGB
test_loader = DataLoader(test_dataset, batch_size=1)

# Output dir
os.makedirs("outputs/ct", exist_ok=True)

# Run inference
with torch.no_grad():
    for i, (noisy, _) in enumerate(test_loader):
        noisy = noisy.to(device)
        output = model(noisy)
        save_image(output, f"outputs/ct/output_{i+1}.png")
        if i >= 4: break  # Save only 5 samples