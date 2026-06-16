import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import OCRDataset
from model import CRNN
from utils import CHARS, decode_ctc

WEIGHTS_CANDIDATES = [
    "best_crnn_iiit5k_alnum_acc.pth",
    "best_crnn_iiit5k_alnum_loss.pth",
    "best_crnn_iiit5k_mix_acc.pth",
    "best_crnn_iiit5k_mix_loss.pth",
    "best_crnn_iiit5k_aug_acc.pth",
    "best_crnn_iiit5k_aug.pth",
    "best_crnn_iiit5k.pth",
]

TEST_DIR_CANDIDATES = [
    "iiit5k_alnum/test",
    "iiit5k_easy/test",
]

def levenshtein(a: str, b: str) -> int:
    n, m = len(a), len(b)

    if n == 0:
        return m
    if m == 0:
        return n

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[n][m]

def collate_fn(batch):
    images, texts = zip(*batch)
    images = torch.stack(images, dim=0)
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

def choose_test_dir():
    for path in TEST_DIR_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("No test dir found.")

def error_type(gt: str, pred: str) -> str:
    if pred == gt:
        return "correct"

    if len(pred) > len(gt):
        return "too_long"
    if len(pred) < len(gt):
        return "too_short"

    return "char_confusion"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    weights_path = choose_weights()
    test_dir = choose_test_dir()

    print("using weights:", weights_path)
    print("using test dir:", test_dir)

    transform = transforms.Compose([
        transforms.Resize((48, 120)),
        transforms.ToTensor(),
    ])

    test_dataset = OCRDataset(test_dir, transform=transform)
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

    results = []
    stats = {
        "correct": 0,
        "too_long": 0,
        "too_short": 0,
        "char_confusion": 0,
    }

    with torch.no_grad():
        for batch_idx, (images, texts) in enumerate(test_loader):
            print(f"batch {batch_idx + 1}/{len(test_loader)}")

            images = images.to(device)
            logits = model(images)
            preds = greedy_decode(logits)

            for gt, pred in zip(texts, preds):
                gt = gt.strip()
                pred = pred.strip()

                dist = levenshtein(gt, pred)
                kind = error_type(gt, pred)

                stats[kind] += 1

                results.append({
                    "gt": gt,
                    "pred": pred,
                    "distance": dist,
                    "gt_len": len(gt),
                    "pred_len": len(pred),
                    "type": kind,
                })

    total = len(results)
    correct = stats["correct"]
    acc = correct / total if total > 0 else 0.0

    hardest = sorted(
        results,
        key=lambda x: (
            x["distance"],
            abs(x["gt_len"] - x["pred_len"]),
            x["gt_len"]
        ),
        reverse=True
    )

    print("\n" + "=" * 60)
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {acc:.4f}")
    print("-" * 60)
    print("Error type stats:")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print("=" * 60)

    print("\n===== Top 30 Hardest Samples =====")
    for i, item in enumerate(hardest[:30], 1):
        print(
            f"{i:02d}. "
            f"[{item['type']}] "
            f"dist={item['distance']} | "
            f"GT: {item['gt']} | Pred: {item['pred']}"
        )

    print("\n===== Top 20 Too Long Errors =====")
    too_long = [x for x in results if x["type"] == "too_long"]
    too_long = sorted(too_long, key=lambda x: x["distance"], reverse=True)
    for i, item in enumerate(too_long[:20], 1):
        print(f"{i:02d}. GT: {item['gt']} | Pred: {item['pred']} | dist={item['distance']}")

    print("\n===== Top 20 Too Short Errors =====")
    too_short = [x for x in results if x["type"] == "too_short"]
    too_short = sorted(too_short, key=lambda x: x["distance"], reverse=True)
    for i, item in enumerate(too_short[:20], 1):
        print(f"{i:02d}. GT: {item['gt']} | Pred: {item['pred']} | dist={item['distance']}")

    print("\n===== Top 20 Character Confusions =====")
    confusion = [x for x in results if x["type"] == "char_confusion"]
    confusion = sorted(confusion, key=lambda x: x["distance"], reverse=True)
    for i, item in enumerate(confusion[:20], 1):
        print(f"{i:02d}. GT: {item['gt']} | Pred: {item['pred']} | dist={item['distance']}")

if __name__ == "__main__":
    main()
