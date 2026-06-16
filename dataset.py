import os
from PIL import Image
from torch.utils.data import Dataset

from utils import CHARS

def clean_label(text: str) -> str:
    allowed = set(CHARS)
    return "".join([c for c in text.strip() if c in allowed])

class OCRDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")
        self.label_file = os.path.join(root_dir, "labels.txt")
        self.transform = transform

        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not os.path.exists(self.label_file):
            raise FileNotFoundError(f"Label file not found: {self.label_file}")

        self.samples = []
        with open(self.label_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                filename, text = line.split(maxsplit=1)
                text = clean_label(text)

                if len(text) == 0:
                    continue

                image_path = os.path.join(self.image_dir, filename)
                if not os.path.exists(image_path):
                    continue

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
