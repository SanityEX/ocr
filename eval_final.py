import os
import argparse
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model_attention_ocr import AttentionOCR
from utils import (
    VOCAB_SIZE,
    SOS_IDX, EOS_IDX,
    decode_attention
)

IMG_H = 48
IMG_W = 192
MAX_LABEL_LEN = 25


class TxtOCRDataset(Dataset):
    def __init__(self, txt_path, img_dir, transform=None):
        self.samples = []
        self.transform = transform

        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if "\t" in line:
                    path, label = line.split("\t", 1)
                else:
                    path, label = line.split(maxsplit=1)

                path = os.path.join(img_dir, os.path.basename(path))
                path = path.replace("\\", "/")

                if os.path.exists(path):
                    self.samples.append((path, label.strip().upper()))

        print(f"loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")

        if self.transform:
            img = self.transform(img)

        return img, label


def val_transform():
    return transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ToTensor(),
    ])


def clean_decode(seq):
    result = []
    for idx in seq:
        if idx == EOS_IDX:
            break
        if idx >= 3:
            result.append(idx)
    return decode_attention(result)


@torch.no_grad()
def greedy_decode(model, image, device):
    image = image.unsqueeze(0).to(device)

    pred_tokens = model.predict(
        image,
        SOS_IDX,
        EOS_IDX,
        max_len=MAX_LABEL_LEN
    )

    seq = pred_tokens.squeeze(0).cpu().tolist()
    return clean_decode(seq)


@torch.no_grad()
def beam_decode(model, image, device, beam_size=5):
    image = image.unsqueeze(0).to(device)

    pred_tokens = model.predict_beam(
        image,
        SOS_IDX,
        EOS_IDX,
        beam_size=beam_size,
        max_len=MAX_LABEL_LEN
    )

    seq = pred_tokens.squeeze(0).cpu().tolist()
    return clean_decode(seq)


def load_model(weights, device):
    model = AttentionOCR(vocab_size=VOCAB_SIZE).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, loader, device, beam_size=5):
    total = 0
    correct_g = 0
    correct_b = 0

    for i, (images, labels) in enumerate(loader):
        for j in range(images.size(0)):
            img = images[j]
            gt = labels[j].strip().upper()

            pred_g = greedy_decode(model, img, device)
            pred_b = beam_decode(model, img, device, beam_size)

            if pred_g.strip().upper() == gt:
                correct_g += 1

            if pred_b.strip().upper() == gt:
                correct_b += 1

            total += 1

            if total % 200 == 0:
                print(
                    f"[{total}] "
                    f"greedy={correct_g/total:.4f} "
                    f"beam={correct_b/total:.4f}"
                )

    print("=" * 50)
    print(f"Total: {total}")
    print(f"Greedy Accuracy: {correct_g/total:.4f}")
    print(f"Beam Accuracy:   {correct_b/total:.4f}")
    print(f"Gain: {correct_b/total - correct_g/total:+.4f}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--beam_size", type=int, default=5)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    dataset = TxtOCRDataset(
        args.labels,
        args.img_dir,
        transform=val_transform()
    )

    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = load_model(args.weights, device)

    evaluate(model, loader, device, args.beam_size)


if __name__ == "__main__":
    main()