import os
import csv
import torch
from PIL import Image
from torchvision import transforms

from model_attention_ocr import AttentionOCR
from utils import (
    VOCAB_SIZE,
    SOS_IDX,
    EOS_IDX,
    decode_attention
)


ROOT = r"D:\mnist_project\ocr1\recognition\recognition"

LEVEL = "30"

GT_TXT = r"D:\mnist_project\ocr1\recognition\recognition\gt_recognition.txt"

IMG_DIR = os.path.join(ROOT, LEVEL)

WEIGHTS = r"D:\mnist_project\ocr1\best_attention_qc_occlusion_v2_acc.pth"

IMG_H = 48
IMG_W = 192
MAX_LEN = 25

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
])


def load_gt(path):
    data = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split(",")

            if len(parts) < 2:
                continue

            name = parts[0].strip()

            label = ",".join(parts[1:]).strip()

            label = label.replace('"', "")
            label = label.upper()

            data[name] = label

    return data


def clean_decode(seq):
    result = []

    for idx in seq:
        if idx == EOS_IDX:
            break

        if idx >= 3:
            result.append(idx)

    return decode_attention(result).strip().upper()


@torch.no_grad()
def predict(model, image_path):
    img = Image.open(image_path).convert("L")

    x = transform(img).unsqueeze(0).to(DEVICE)

    pred = model.predict(
        x,
        SOS_IDX,
        EOS_IDX,
        max_len=MAX_LEN
    )

    pred = pred.squeeze(0).cpu().tolist()

    return clean_decode(pred)


def main():

    print("device:", DEVICE)
    print("level:", LEVEL)

    gt_data = load_gt(GT_TXT)

    model = AttentionOCR(
        vocab_size=VOCAB_SIZE
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            WEIGHTS,
            map_location=DEVICE
        )
    )

    model.eval()

    total = 0
    correct = 0

    wrong_cases = []

    image_files = sorted(os.listdir(IMG_DIR))

    for i, filename in enumerate(image_files):

        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        gt_name = filename.replace("img_", "word_")

        gt_name = os.path.splitext(gt_name)[0] + ".png"

        if gt_name not in gt_data:
            continue

        gt = gt_data[gt_name]

        image_path = os.path.join(
            IMG_DIR,
            filename
        )

        pred = predict(model, image_path)

        total += 1

        ok = pred == gt

        if ok:
            correct += 1
        else:
            if len(wrong_cases) < 30:
                wrong_cases.append(
                    (filename, gt, pred)
                )

        if total % 200 == 0:
            print(
                f"[{total}] "
                f"acc={correct/total:.4f}"
            )

    print("=" * 60)
    print("level:", LEVEL)
    print("total:", total)
    print("correct:", correct)
    print("accuracy:", correct / total)
    print("=" * 60)

    print("\nWrong examples:\n")

    for name, gt, pred in wrong_cases:
        print(
            f"{name} | "
            f"GT={gt} | "
            f"PRED={pred}"
        )


if __name__ == "__main__":
    main()