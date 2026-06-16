import os
import torch
from PIL import Image
from torchvision import transforms


ROOT = r"D:\mnist_project\ocr1\recognition\recognition"
GT_TXT = r"D:\mnist_project\ocr1\recognition\recognition\gt_recognition.txt"

LEVEL = "20"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_H = 32
IMG_W = 128


transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


def normalize_text(text):
    return "".join(ch for ch in text.upper() if ch.isalnum())


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
            data[name] = label

    return data


@torch.no_grad()
def predict_one(model, img_path):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    logits = model(x)
    probs = logits.softmax(-1)

    labels, confidences = model.tokenizer.decode(probs)

    pred = labels[0]
    conf = float(confidences[0].mean().item()) if len(confidences[0]) > 0 else 0.0

    return normalize_text(pred), conf


def main():
    print("device:", DEVICE)
    print("level:", LEVEL)

    model = torch.hub.load(
        "baudm/parseq",
        "parseq",
        pretrained=True
    ).to(DEVICE)

    model.eval()

    gt_data = load_gt(GT_TXT)

    img_dir = os.path.join(ROOT, LEVEL)

    total = 0
    correct = 0

    wrong_cases = []

    image_files = sorted(os.listdir(img_dir))

    for filename in image_files:
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        gt_name = filename.replace("img_", "word_")
        gt_name = os.path.splitext(gt_name)[0] + ".png"

        if gt_name not in gt_data:
            continue

        gt = normalize_text(gt_data[gt_name])

        if not gt:
            continue

        img_path = os.path.join(img_dir, filename)

        pred, conf = predict_one(model, img_path)

        total += 1

        if pred == gt:
            correct += 1
        else:
            if len(wrong_cases) < 30:
                wrong_cases.append((filename, gt, pred, conf))

        if total % 200 == 0:
            print(f"[{total}] acc={correct / total:.4f}")

    print("=" * 60)
    print("PARSeq Result")
    print("level:", LEVEL)
    print("total:", total)
    print("correct:", correct)
    print("accuracy:", correct / total)
    print("=" * 60)

    print("\nWrong examples:\n")

    for name, gt, pred, conf in wrong_cases:
        print(f"{name} | GT={gt} | PRED={pred} | conf={conf:.4f}")


if __name__ == "__main__":
    main()