import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class OCRDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")
        self.label_file = os.path.join(root_dir, "labels.txt")
        self.transform = transform

        self.samples = []
        with open(self.label_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                filename, text = line.split(maxsplit=1)
                self.samples.append((filename, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, text = self.samples[idx]
        image_path = os.path.join(self.image_dir, filename)

        image = Image.open(image_path).convert("L")

        if self.transform is not None:
            image = self.transform(image)

        return image, text