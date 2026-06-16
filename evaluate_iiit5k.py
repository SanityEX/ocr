import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import OCRDataset
from model import CRNN
from utils import CHARS, decode_ctc


WEIGHTS_CANDIDATES = [
    "best_crnn_iiit5k_dyn_acc.pth",
    "best_crnn_iiit5k_dyn_loss.pth",
    "best_crnn_iiit5k_alnum_acc.pth",
    "best_crnn_iiit5k_mix_acc.pth",
]

TEST_DIR = "iiit5k_alnum/test"

IMG_H = 48
MIN_W = 48
MAX_W = 220


def resize_keep_ratio(image, img_h=IMG_H, min_w=MIN_W, max_w=MAX_W):
    w, h = image.size
    new_w = max(1, int(w * img_h / h))
    new_w = max(min_w, min(new_w, max_w))
    image = image.resize((new_w, img_h))
    return image


def collate_fn(batch):
    to_tensor = transforms.ToTensor()

    images_pil, texts = zip(*batch)
    processed = []
    widths = []

    for image in images_pil:
        image = resize_keep_ratio(image)
        tensor = to_tensor(image)
        processed.append(tensor)
        widths.append(tensor.shape[-1])

    max_width = max(widths)

    padded_images = []
    for tensor in processed:
        c, h, w = tensor.shape
        pad_w = max_width - w
        padded = F.pad(tensor, (0, pad_w, 0, 0), value=0.0)
        padded_images.append(padded)

    images = torch.stack(padded_images, dim=0)
    return images, texts


@torch.no_grad()
def greedy_decode(logits: torch.Tensor) -> list[str]:
    preds = logits.argmax(dim=2)
    preds = preds.permute(1, 0)

    results = []
    for seq in preds:
        indices = []
        prev = None
        for idx in seq.cpu().tolist():
            if idx != 0 and idx != prev:
                indices.append(idx)
            prev = idx
        results.append(decode_ctc(indices))
    return results


def choose_weights():
    for path in WEIGHTS_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("No weights found.")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    weights_path = choose_weights()
    print("using weights:", weights_path)

    test_dataset = OCRDataset(TEST_DIR, transform=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn
    )

    print("test samples:", len(test_dataset))

    num_classes = len(CHARS) + 1
    model = CRNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    correct = 0
    total = 0
    shown = 0

    with torch.no_grad():
        for batch_idx, (images, texts) in enumerate(test_loader):
            print(f"batch {batch_idx + 1}/{len(test_loader)}")

            images = images.to(device)
            logits = model(images)
            preds = greedy_decode(logits)

            for gt, pred in zip(texts, preds):
                if shown < 50:
                    print(f"GT: {gt} | Pred: {pred}")
                    shown += 1

                if pred == gt:
                    correct += 1
                total += 1

    acc = correct / total if total > 0 else 0.0

    print("\n" + "=" * 50)
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {acc:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    if not os.path.exists(TEST_DIR):
        raise FileNotFoundError(f"Test dir not found: {TEST_DIR}")
    main()