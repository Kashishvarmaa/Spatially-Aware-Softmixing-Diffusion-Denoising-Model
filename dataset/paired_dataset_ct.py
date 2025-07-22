
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class PairedImageDataset_ct(Dataset):
    def __init__(self, root_dir, image_size=(256, 256), as_gray=False):
        super().__init__()
        self.clean_dir = os.path.join(root_dir, 'clean')
        self.noisy_dir = os.path.join(root_dir, 'noisy')

        self.image_size = image_size
        self.clean_files = sorted(os.listdir(self.clean_dir))
        self.noisy_files = sorted(os.listdir(self.noisy_dir))
        self.as_gray = as_gray

        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        mode = 'L' if self.as_gray else 'RGB'

        clean_img = Image.open(os.path.join(self.clean_dir, self.clean_files[idx])).convert(mode)
        noisy_img = Image.open(os.path.join(self.noisy_dir, self.noisy_files[idx])).convert(mode)

        clean = self.transform(clean_img)
        noisy = self.transform(noisy_img)

        return noisy, clean