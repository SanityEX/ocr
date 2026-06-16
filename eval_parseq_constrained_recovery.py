import os
import difflib
from collections import Counter

import torch
from PIL import Image
from torchvision import transforms

ROOT = r"D:\mnist_project\ocr1\recognition\recognition"
GT_TXT = r"D:\mnist_project\ocr1\recognition\recognition\gt_recognition.txt"

LEVELS = ["10", "20", "30"]

IMG_H = 32
IMG_W = 128

LOW_CONF_TH = 0.85
MAX_EDIT_DIST = 3
MAX_LEN_DIFF = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def build_lexicon(gt_data):
    words = set()

    for label in gt_data.values():
        word = normalize_text(label)

        if 2 <= len(word) <= 25:
            words.add(word)

    return sorted(words)

def levenshtein(a, b):
    n = len(a)
    m = len(b)

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

def char_overlap(a, b):
    ca = Counter(a)
    cb = Counter(b)

    common = 0

    for ch in ca:
        common += min(ca[ch], cb.get(ch, 0))

    return common / max(len(a), len(b), 1)

def fixed_position_match(word, pred, low_positions):
    if abs(len(word) - len(pred)) > MAX_LEN_DIFF:
        return False

    if len(word) == len(pred):
        for i, ch in enumerate(pred):
            if i not in low_positions:
                if word[i] != ch:
                    return False

        return True

    matcher = difflib.SequenceMatcher(None, pred, word)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        affected_pred_positions = set(range(i1, i2))

        if not affected_pred_positions:
            continue

        for pos in affected_pred_positions:
            if pos not in low_positions:
                return False

    return True

def constrained_recover(pred, conf_list, lexicon):
    if not pred:
        return pred, []

    low_positions = set()

    for i, c in enumerate(conf_list[:len(pred)]):
        if c < LOW_CONF_TH:
            low_positions.add(i)

    if not low_positions:
        return pred, []

    candidates = []

    for word in lexicon:
        if abs(len(word) - len(pred)) > MAX_LEN_DIFF:
            continue

        edit = levenshtein(word, pred)

        if edit > MAX_EDIT_DIST:
            continue

        if not fixed_position_match(word, pred, low_positions):
            continue

        overlap = char_overlap(word, pred)

        len_penalty = abs(len(word) - len(pred)) * 0.3

        score = (
            -edit
            + overlap * 2.0
            - len_penalty
        )

        candidates.append(
            (
                word,
                score,
                edit,
                overlap,
                sorted(list(low_positions))
            )
        )

    if not candidates:
        return pred, []

    candidates.sort(key=lambda x: x[1], reverse=True)

    best = candidates[0][0]

    return best, candidates[:5]

def detect_errors(gt, pred):
    matcher = difflib.SequenceMatcher(None, gt, pred)

    errors = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            errors.append(("replace", gt[i1:i2], pred[j1:j2]))

        elif tag == "delete":
            errors.append(("missing", gt[i1:i2], ""))

        elif tag == "insert":
            errors.append(("extra", "", pred[j1:j2]))

    return errors

class ParseqConstrainedOCR:
    def __init__(self, device):
        self.device = device

        self.model = torch.hub.load(
            "baudm/parseq",
            "parseq",
            pretrained=True,
            trust_repo=True
        ).to(device)

        self.model.eval()

    @torch.no_grad()
    def predict(self, image_path):
        img = Image.open(image_path).convert("RGB")

        x = transform(img).unsqueeze(0).to(self.device)

        logits = self.model(x)

        probs = logits.softmax(-1)

        labels, confidences = self.model.tokenizer.decode(probs)

        pred = normalize_text(labels[0])

        conf_list = []

        if len(confidences[0]) > 0:
            for c in confidences[0]:
                conf_list.append(float(c.item()))

        avg_conf = sum(conf_list) / len(conf_list) if conf_list else 0.0

        return pred, conf_list, avg_conf

def evaluate_level(model, gt_data, lexicon, level):
    img_dir = os.path.join(ROOT, level)

    total = 0
    raw_correct = 0
    constrained_correct = 0

    changed_count = 0
    improved_count = 0
    damaged_count = 0

    wrong_cases = []

    for filename in sorted(os.listdir(img_dir)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        gt_name = filename.replace("img_", "word_")
        gt_name = os.path.splitext(gt_name)[0] + ".png"

        if gt_name not in gt_data:
            continue

        gt = normalize_text(gt_data[gt_name])

        if not gt:
            continue

        image_path = os.path.join(img_dir, filename)

        pred, conf_list, avg_conf = model.predict(image_path)

        recovered, candidates = constrained_recover(
            pred,
            conf_list,
            lexicon
        )

        total += 1

        raw_ok = pred == gt
        rec_ok = recovered == gt

        if raw_ok:
            raw_correct += 1

        if rec_ok:
            constrained_correct += 1

        if recovered != pred:
            changed_count += 1

            if not raw_ok and rec_ok:
                improved_count += 1

            if raw_ok and not rec_ok:
                damaged_count += 1

        if not rec_ok and len(wrong_cases) < 25:
            errors = detect_errors(gt, recovered)

            low_positions = [
                (i, c)
                for i, c in enumerate(conf_list[:len(pred)])
                if c < LOW_CONF_TH
            ]

            wrong_cases.append({
                "file": filename,
                "gt": gt,
                "pred": pred,
                "recovered": recovered,
                "avg_conf": avg_conf,
                "low_positions": low_positions,
                "errors": errors,
                "candidates": candidates
            })

        if total % 200 == 0:
            print(
                f"[level {level}] [{total}] "
                f"raw={raw_correct / total:.4f} "
                f"constrained={constrained_correct / total:.4f} "
                f"changed={changed_count}"
            )

    raw_acc = raw_correct / total
    constrained_acc = constrained_correct / total

    return {
        "total": total,
        "raw_correct": raw_correct,
        "constrained_correct": constrained_correct,
        "raw_acc": raw_acc,
        "constrained_acc": constrained_acc,
        "gain": constrained_acc - raw_acc,
        "changed_count": changed_count,
        "improved_count": improved_count,
        "damaged_count": damaged_count,
        "wrong_cases": wrong_cases
    }

def main():
    print("device:", DEVICE)
    print("low confidence threshold:", LOW_CONF_TH)

    gt_data = load_gt(GT_TXT)

    lexicon = build_lexicon(gt_data)

    print("lexicon size:", len(lexicon))

    model = ParseqConstrainedOCR(DEVICE)

    print("=" * 70)
    print("PARSeq + Constrained Character-level Recovery")
    print("=" * 70)

    for level in LEVELS:
        print("\n" + "=" * 70)
        print("LEVEL:", level)

        result = evaluate_level(
            model,
            gt_data,
            lexicon,
            level
        )

        print("-" * 70)
        print("total:", result["total"])
        print("raw correct:", result["raw_correct"])
        print("constrained correct:", result["constrained_correct"])
        print("raw acc:", result["raw_acc"])
        print("constrained acc:", result["constrained_acc"])
        print("gain:", result["gain"])
        print("changed:", result["changed_count"])
        print("improved:", result["improved_count"])
        print("damaged:", result["damaged_count"])
        print("-" * 70)

        print("Wrong examples:")

        for case in result["wrong_cases"]:
            print(
                f"{case['file']} | "
                f"GT={case['gt']} | "
                f"PARSeq={case['pred']} | "
                f"Constrained={case['recovered']} | "
                f"conf={case['avg_conf']:.4f}"
            )

            if case["low_positions"]:
                print("  low positions:", case["low_positions"])

            if case["errors"]:
                print("  errors:", case["errors"])

            if case["candidates"]:
                print("  candidates:")
                for word, score, edit, overlap, low_pos in case["candidates"]:
                    print(
                        f"    {word:20s} "
                        f"score={score:.4f} "
                        f"edit={edit} "
                        f"overlap={overlap:.2f} "
                        f"low={low_pos}"
                    )

    print("\nDONE")

if __name__ == "__main__":
    main()
