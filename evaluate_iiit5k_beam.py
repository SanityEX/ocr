import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import OCRDataset
from model import CRNN
from utils import CHARS
from decode_utils import greedy_decode_from_logits, ctc_beam_search_batch


WEIGHTS_PATH = "best_crnn_iiit5k.pth"
TEST_DIR = "iiit5k_easy/test"
BEAM_WIDTH = 10


def collate_fn(batch):
    images, texts = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, texts


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    transform = transforms.Compose([
        transforms.Resize((48, 120)),
        transforms.ToTensor(),
    ])

    test_dataset = OCRDataset(TEST_DIR, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn
    )

    num_classes = len(CHARS) + 1
    model = CRNN(num_classes=num_classes).to(device)

    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"weights not found: {WEIGHTS_PATH}")

    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()

    greedy_correct = 0
    beam_correct = 0
    total = 0

    shown = 0

    with torch.no_grad():
        for batch_idx, (images, texts) in enumerate(test_loader):
            print(f"batch {batch_idx + 1}/{len(test_loader)}")

            images = images.to(device)
            logits = model(images)

            greedy_preds = greedy_decode_from_logits(logits)
            beam_preds = ctc_beam_search_batch(logits, beam_width=BEAM_WIDTH)

            for gt, g_pred, b_pred in zip(texts, greedy_preds, beam_preds):
                gt = gt.lower().strip()
                g_pred = g_pred.lower().strip()
                b_pred = b_pred.lower().strip()

                if g_pred == gt:
                    greedy_correct += 1
                if b_pred == gt:
                    beam_correct += 1

                if shown < 40:
                    print(f"GT: {gt} | Greedy: {g_pred} | Beam: {b_pred}")
                    shown += 1

                total += 1

    greedy_acc = greedy_correct / total if total > 0 else 0.0
    beam_acc = beam_correct / total if total > 0 else 0.0

    print("\n" + "=" * 60)
    print(f"Total: {total}")
    print(f"Greedy correct: {greedy_correct}")
    print(f"Greedy accuracy: {greedy_acc:.4f}")
    print(f"Beam correct:   {beam_correct}")
    print(f"Beam accuracy:  {beam_acc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()