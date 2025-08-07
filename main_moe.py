import torch
from PIL import Image
from torchvision import transforms

from experts.unet_raindrop import UNet as RaindropUNet
from experts.unet_ct import UNet as CTUNet
from enhancer.diffusion_refined import DiffusionRefiner
from gating.load_gate import predict_class
from utils.image_utils import save_image

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def load_expert(label):
    if label == 0:  # medical
        model = CTUNet(in_channels=1, out_channels=1)
        checkpoint = "experts/checkpoints/ct/epoch_100.pth"
    else:  # raindrop
        model = RaindropUNet(in_channels=3, out_channels=3)
        checkpoint = "experts/checkpoints/raindrop/epoch_100.pth"

    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

def moe_predict(image_pil, return_steps=False):
    # Classify first
    label = predict_class(image_pil)

    # Preprocess according to class
    if label == 0:  # medical/CT
        image_pil = image_pil.convert('L')
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        img_tensor = transform(image_pil)  # [1, H, W]
    else:  # raindrop
        image_pil = image_pil.convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        img_tensor = transform(image_pil)  # [3, H, W]

    expert = load_expert(label)
    input_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = expert(input_tensor)

    denoised_tensor = output.squeeze(0).cpu()
    from torchvision.transforms.functional import to_pil_image
    denoised_pil = to_pil_image(denoised_tensor)

    refiner = DiffusionRefiner()
    enhanced_pil = refiner.enhance(denoised_pil)

    if return_steps:
        return label, denoised_pil, enhanced_pil
    return enhanced_pil


