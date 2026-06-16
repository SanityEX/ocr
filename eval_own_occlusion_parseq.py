import os
import math
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
MAX_LEN = 25

USE_LEXICON_RECOVERY = True
MAX_EDIT_DIST = 3

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

    for v in gt_data.values():
        w = normalize_text(v)

        if 2 <= len(w) <= MAX_LEN:
            words.add(w)

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


def detect_error_type(gt, pred):
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


class OwnOcclusionPARSeq:
    def __init__(self, device):
        self.device = device

        self.backbone = torch.hub.load(
            "baudm/parseq",
            "parseq",
            pretrained=True,
            trust_repo=True
        ).to(device)

        self.backbone.eval()

    @torch.no_grad()
    def predict(self, image_path):
        img = Image.open(image_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(self.device)

        logits = self.backbone(x)
        probs = logits.softmax(-1)

        labels, confidences = self.backbone.tokenizer.decode(probs)

        pred = normalize_text(labels[0])

        conf_list = []

        if len(confidences[0]) > 0:
            for c in confidences[0]:
                conf_list.append(float(c.item()))

        avg_conf = sum(conf_list) / len(conf_list) if conf_list else 0.0

        uncertain_positions = []

        for i, c in enumerate(conf_list):
            if c < LOW_CONF_TH:
                uncertain_positions.append((i, c))

        return {
            "pred": pred,
            "avg_conf": avg_conf,
            "char_conf": conf_list,
            "uncertain_positions": uncertain_positions
        }

    def recover_with_lexicon(self, pred, lexicon):
        if not USE_LEXICON_RECOVERY:
            return pred, []

        candidates = []

        for word in lexicon:
            if abs(len(word) - len(pred)) > 3:
                continue

            edit = levenshtein(word, pred)

            if edit > MAX_EDIT_DIST:
                continue

            overlap = char_overlap(word, pred)

            score = -edit + overlap * 2.0

            candidates.append((word, score, edit, overlap))

        if not candidates:
            return pred, []

        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates[0][0], candidates[:5]


def evaluate_level(model, gt_data, lexicon, level):
    img_dir = os.path.join(ROOT, level)

    total = 0
    raw_correct = 0
    recovered_correct = 0

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

        result = model.predict(image_path)

        pred = result["pred"]
        recovered, candidates = model.recover_with_lexicon(pred, lexicon)

        total += 1

        if pred == gt:
            raw_correct += 1

        if recovered == gt:
            recovered_correct += 1
        else:
            if len(wrong_cases) < 25:
                errors = detect_error_type(gt, recovered)

                wrong_cases.append({
                    "file": filename,
                    "gt": gt,
                    "pred": pred,
                    "recovered": recovered,
                    "conf": result["avg_conf"],
                    "uncertain": result["uncertain_positions"],
                    "errors": errors,
                    "candidates": candidates
                })

        if total % 200 == 0:
            print(
                f"[level {level}] [{total}] "
                f"raw={raw_correct / total:.4f} "
                f"own_model={recovered_correct / total:.4f}"
            )

    raw_acc = raw_correct / total
    recovered_acc = recovered_correct / total

    return total, raw_correct, recovered_correct, raw_acc, recovered_acc, wrong_cases


def main():
    print("device:", DEVICE)

    gt_data = load_gt(GT_TXT)
    lexicon = build_lexicon(gt_data)

    print("lexicon size:", len(lexicon))
    print("use lexicon recovery:", USE_LEXICON_RECOVERY)

    model = OwnOcclusionPARSeq(DEVICE)

    print("=" * 70)
    print("Own Occlusion-aware PARSeq Model")
    print("=" * 70)

    for level in LEVELS:
        print("\n" + "=" * 70)
        print("LEVEL:", level)

        total, raw_correct, recovered_correct, raw_acc, recovered_acc, wrong_cases = evaluate_level(
            model,
            gt_data,
            lexicon,
            level
        )

        print("-" * 70)
        print("total:", total)
        print("PARSeq raw correct:", raw_correct)
        print("Own model correct:", recovered_correct)
        print("PARSeq raw acc:", raw_acc)
        print("Own model acc:", recovered_acc)
        print("gain:", recovered_acc - raw_acc)
        print("-" * 70)

        print("Wrong examples:")

        for case in wrong_cases:
            print(
                f"{case['file']} | "
                f"GT={case['gt']} | "
                f"PARSeq={case['pred']} | "
                f"Own={case['recovered']} | "
                f"conf={case['conf']:.4f}"
            )

            if case["uncertain"]:
                print("  uncertain:", case["uncertain"])

            if case["errors"]:
                print("  errors:", case["errors"])

            if case["candidates"]:
                print("  candidates:")
                for w, score, edit, overlap in case["candidates"]:
                    print(
                        f"    {w:20s} "
                        f"score={score:.4f} "
                        f"edit={edit} "
                        f"overlap={overlap:.2f}"
                    )

    print("\nDONE")


if __name__ == "__main__":
    main()