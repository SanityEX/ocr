import os
import csv
from collections import defaultdict

import torch
from PIL import Image
from torchvision import transforms

from model_attention_ocr import AttentionOCR
from utils import VOCAB_SIZE, SOS_IDX, EOS_IDX, decode_attention

LABELS = r"D:\mnist_project\ocr1\word_occluded_10k\labels.txt"
IMG_DIR = r"D:\mnist_project\ocr1\word_occluded_10k\images"
META_CSV = r"D:\mnist_project\ocr1\word_occluded_10k\metadata.csv"
WEIGHTS = r"D:\mnist_project\ocr1\best_attention_v2_phase2_acc.pth"

OUT_TXT = "occluded_analysis_result.txt"

IMG_H = 48
IMG_W = 192
MAX_LEN = 25

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
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
    return decode_attention(result).strip().upper()

def load_labels(path):
    data = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if "\t" in line:
                name, label = line.split("\t", 1)
            else:
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                name, label = parts

            data[os.path.basename(name)] = label.strip().upper()

    return data

def load_meta(path):
    data = {}

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            name = row["filename"]
            data[name] = row

    return data

@torch.no_grad()
def predict_one(model, img_path):
    img = Image.open(img_path).convert("L")
    x = transform(img).unsqueeze(0).to(DEVICE)

    pred = model.predict(
        x,
        SOS_IDX,
        EOS_IDX,
        max_len=MAX_LEN
    )

    return clean_decode(pred.squeeze(0).cpu().tolist())

def update_stat(stat, key, ok):
    stat[key]["total"] += 1
    if ok:
        stat[key]["correct"] += 1

def acc(item):
    if item["total"] == 0:
        return 0
    return item["correct"] / item["total"]

def format_stat(title, stat):
    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append(title)
    lines.append("=" * 60)

    for key, v in sorted(stat.items(), key=lambda x: str(x[0])):
        lines.append(
            f"{key:15s}  total={v['total']:5d}  correct={v['correct']:5d}  acc={acc(v):.4f}"
        )

    return lines

def main():
    print("device:", DEVICE)

    labels = load_labels(LABELS)
    meta = load_meta(META_CSV)

    model = AttentionOCR(vocab_size=VOCAB_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
    model.eval()

    stat_type = defaultdict(lambda: {"total": 0, "correct": 0})
    stat_direction = defaultdict(lambda: {"total": 0, "correct": 0})
    stat_ratio = defaultdict(lambda: {"total": 0, "correct": 0})
    stat_char = defaultdict(lambda: {"total": 0, "correct": 0})
    stat_length = defaultdict(lambda: {"total": 0, "correct": 0})

    wrong_cases = []

    total = 0
    correct = 0

    filenames = list(labels.keys())

    for i, name in enumerate(filenames):
        img_path = os.path.join(IMG_DIR, name)

        if not os.path.exists(img_path):
            continue

        gt = labels[name]
        pred = predict_one(model, img_path)

        ok = pred == gt

        total += 1
        if ok:
            correct += 1

        m = meta.get(name, {})

        occ_type = m.get("occlusion_type", "unknown")
        direction = m.get("direction", "unknown")
        ratio = m.get("ratio", "unknown")
        occ_char = m.get("occluded_char", "unknown").upper()

        if len(gt) <= 5:
            length_group = "short"
        elif len(gt) <= 9:
            length_group = "middle"
        else:
            length_group = "long"

        update_stat(stat_type, occ_type, ok)
        update_stat(stat_direction, direction, ok)
        update_stat(stat_ratio, str(ratio), ok)
        update_stat(stat_char, occ_char, ok)
        update_stat(stat_length, length_group, ok)

        if not ok and len(wrong_cases) < 50:
            wrong_cases.append((name, gt, pred, occ_type, direction, ratio, occ_char))

        if (i + 1) % 500 == 0:
            print(f"[{i+1}/{len(filenames)}] acc={correct/total:.4f}")

    lines = []

    lines.append("Occluded OCR Analysis")
    lines.append("=" * 60)
    lines.append(f"Total:   {total}")
    lines.append(f"Correct: {correct}")
    lines.append(f"Accuracy:{correct / total:.4f}")

    lines += format_stat("Accuracy by occlusion type", stat_type)
    lines += format_stat("Accuracy by direction", stat_direction)
    lines += format_stat("Accuracy by ratio", stat_ratio)
    lines += format_stat("Accuracy by word length", stat_length)

    lines.append("")
    lines.append("=" * 60)
    lines.append("Accuracy by occluded character")
    lines.append("=" * 60)

    sorted_chars = sorted(
        stat_char.items(),
        key=lambda x: x[1]["total"],
        reverse=True
    )

    for key, v in sorted_chars:
        lines.append(
            f"{key:5s}  total={v['total']:5d}  correct={v['correct']:5d}  acc={acc(v):.4f}"
        )

    lines.append("")
    lines.append("=" * 60)
    lines.append("Wrong examples")
    lines.append("=" * 60)

    for name, gt, pred, occ_type, direction, ratio, occ_char in wrong_cases:
        lines.append(
            f"{name} | GT={gt} | Pred={pred} | type={occ_type} | dir={direction} | ratio={ratio} | char={occ_char}"
        )

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("=" * 60)
    print(f"Total: {total}")
    print(f"Accuracy: {correct / total:.4f}")
    print("saved:", OUT_TXT)
    print("=" * 60)

if __name__ == "__main__":
    main()
