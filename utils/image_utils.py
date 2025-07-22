from PIL import Image
import torchvision.transforms as T
import os
import torch

# Resize to standard input (optional)
STANDARD_SIZE = (256, 256)

transform = T.Compose([
    T.Resize(STANDARD_SIZE),
    T.ToTensor(),
])

def load_image(path):
    image = Image.open(path)
    if image.mode not in ["RGB", "L"]:
        image = image.convert("RGB")
    return image

def save_image(pil_img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pil_img.save(path)

def image_to_tensor(pil_img, grayscale=False):
    if grayscale:
        pil_img = pil_img.convert("L")
    else:
        pil_img = pil_img.convert("RGB")
    return transform(pil_img).unsqueeze(0)  # Add batch dim

def tensor_to_image(tensor):
    tensor = tensor.squeeze(0).detach().cpu()
    if tensor.shape[0] == 1:
        tensor = torch.cat([tensor]*3, dim=0)
    return T.ToPILImage()(tensor)