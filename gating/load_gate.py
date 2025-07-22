import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

GATE_PATH = "/Users/kashishvarmaa/Documents/7 Sem/SASM/gating/mobilenet_gate.pth"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Define transform to match training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def load_gate_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model.load_state_dict(torch.load(GATE_PATH, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    return model

def predict_class(image, return_tensor=False):
    """
    Args:
        image (PIL.Image): input image
        return_tensor (bool): if True, also returns the transformed tensor (before batch)

    Returns:
        int: 0 = raindrop, 1 = medical
    """
    model = load_gate_model()
    if image.mode != 'RGB':
        image = image.convert('RGB')

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)  # Add batch dim

    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()

    if return_tensor:
        return transform(image), pred
    return pred