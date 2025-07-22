# enhancer/diffusion_refiner.py

from diffusers import StableDiffusionUpscalePipeline
from PIL import Image
import torch

class DiffusionRefiner:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.pipe = StableDiffusionUpscalePipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler",
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        ).to(self.device)

    def enhance(self, image_pil: Image.Image) -> Image.Image:
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")
        result = self.pipe(prompt="", image=image_pil).images[0]
        return result