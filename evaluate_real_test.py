import os
import torch
from PIL import Image
from torchvision import transforms

from model import CRNN
from utils import CHARS, decode_ctc


WEIGHTS_PATH = "best_crnn_realprint.pth"
REAL_TEST_DIR = "real_test"


def load_labels(label_file):
    samples = []
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            filename, text = line.split(maxsplit=1)
            samples.append((filename, text.lower()))
    return samples


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((48, 120)),
        transforms.ToTensor(),
    ])

    num_classes = len(CHARS) + 1
    model = CRNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()

    image_dir = os.path.join(REAL_TEST_DIR, "images")
    label_file = os.path.join(REAL_TEST_DIR, "labels.txt")

    samples = load_labels(label_file)

    correct = 0
    total = 0

    for filename, gt in samples:
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path).convert("L")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(image)
            pred_indices = logits.argmax(dim=2).squeeze(1).cpu().tolist()
            pred = decode_ctc(pred_indices)

        ok = pred == gt
        print(f"{filename} | GT: {gt} | Pred: {pred} | {'OK' if ok else 'NG'}")

        if ok:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0.0
    print("=" * 40)
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()