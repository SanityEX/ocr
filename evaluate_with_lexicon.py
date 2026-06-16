import os
import torch
import editdistance
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import OCRDataset
from model import CRNN
from utils import decode_ctc, CHARS


WEIGHTS_PATH = "best_crnn_iiit5k.pth"
TEST_DIR = "iiit5k_easy/test"
LEXICON_PATH = "IIIT5K-Word_V3.0/IIIT5K/lexicon.txt"


def collate_fn(batch):
    images, texts = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, texts


@torch.no_grad()
def greedy_decode(logits: torch.Tensor) -> list[str]:
    preds = logits.argmax(dim=2)   # [T, B]
    preds = preds.permute(1, 0)    # [B, T]

    results = []
    for seq in preds:
        results.append(decode_ctc(seq.cpu().tolist()))
    return results


def load_lexicon(path: str) -> list[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"lexicon not found: {path}")

    words = []

    encodings = ["utf-8", "gbk", "latin-1"]

    lines = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                lines = f.readlines()
            print(f"lexicon loaded with encoding: {enc}")
            break
        except UnicodeDecodeError:
            continue

    if lines is None:
        raise UnicodeDecodeError("lexicon", b"", 0, 1, "unable to decode lexicon file")

    for line in lines:
        w = line.strip().lower()
        if w and w.isalpha():
            words.append(w)

    seen = set()
    uniq = []
    for w in words:
        if w not in seen:
            seen.add(w)
            uniq.append(w)

    return uniq


def correct_word_smart(pred: str, lexicon: list[str]) -> str:
    pred = pred.lower().strip()

    if not pred:
        return pred

    if pred in lexicon:
        return pred

    best_word = pred
    best_score = 999

    for word in lexicon:
        if abs(len(word) - len(pred)) > 1:
            continue

        if word[0] != pred[0]:
            continue

        dist = editdistance.eval(pred, word)

        score = dist

        score += abs(len(word) - len(pred)) * 0.5

        if score < best_score:
            best_score = score
            best_word = word

    # 5️⃣ 防止乱改
    if best_score > 2:
        return pred

    return best_word


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
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )

    num_classes = len(CHARS) + 1
    model = CRNN(num_classes=num_classes).to(device)

    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"weights not found: {WEIGHTS_PATH}")

    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()

    lexicon = load_lexicon(LEXICON_PATH)
    print("lexicon size:", len(lexicon))

    raw_correct = 0
    fixed_correct = 0
    total = 0

    print("\n===== sample predictions =====")
    shown = 0

    for images, texts in test_loader:
        images = images.to(device)

        with torch.no_grad():
            logits = model(images)
            preds = greedy_decode(logits)

        for pred, gt in zip(preds, texts):
            gt = gt.lower().strip()
            pred = pred.lower().strip()
            fixed = correct_word_smart(pred, lexicon)

            if pred == gt:
                raw_correct += 1
            if fixed == gt:
                fixed_correct += 1

            if shown < 30:
                print(f"GT: {gt} | Pred: {pred} | Fixed: {fixed}")
                shown += 1

            total += 1

    raw_acc = raw_correct / total if total > 0 else 0.0
    fixed_acc = fixed_correct / total if total > 0 else 0.0

    print("\n" + "=" * 50)
    print(f"Total: {total}")
    print(f"Raw correct:   {raw_correct}")
    print(f"Raw accuracy:  {raw_acc:.4f}")
    print(f"Fixed correct: {fixed_correct}")
    print(f"Fixed accuracy:{fixed_acc:.4f}")


if __name__ == "__main__":
    main()