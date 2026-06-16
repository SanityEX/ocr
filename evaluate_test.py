import os
import random
import torch
from PIL import Image
from torchvision import transforms

from model import CRNN
from utils import CHARS, decode_ctc

def load_labels(label_file):
    samples = []
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            filename, text = line.split(maxsplit=1)
            samples.append((filename, text))
    return samples

def predict_image(model, image_path, transform, device):
    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image)
        pred = logits.argmax(dim=2).squeeze(1).cpu().tolist()
        text = decode_ctc(pred)

    return text

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((32, 120)),
        transforms.ToTensor(),
    ])

    num_classes = len(CHARS) + 1
    model = CRNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("best_crnn.pth", map_location=device))
    model.eval()

    label_file = "data/test/labels.txt"
    image_dir = "data/test/images"

    samples = load_labels(label_file)

    random.seed(42)
    test_samples = random.sample(samples, min(20, len(samples)))

    correct = 0
    total = 0

    print("===== Test Results =====")
    for filename, gt in test_samples:
        image_path = os.path.join(image_dir, filename)
        pred = predict_image(model, image_path, transform, device)

        is_correct = pred == gt
        if is_correct:
            correct += 1
        total += 1

        print(f"{filename} | GT: {gt} | Pred: {pred} | {'OK' if is_correct else 'NG'}")

    acc = correct / total if total > 0 else 0.0
    print("=" * 40)
    print(f"Test samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
