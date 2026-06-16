import os
import math
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model_attention_ocr import AttentionOCR
from utils import (
    VOCAB_SIZE,
    SOS_IDX,
    EOS_IDX,
    idx_to_char
)

ROOT = r"D:\mnist_project\ocr1\recognition\recognition"
GT_TXT = r"D:\mnist_project\ocr1\recognition\recognition\gt_recognition.txt"

LEVEL = "30"

WEIGHTS = r"D:\mnist_project\ocr1\best_attention_qc_occlusion_v2_acc.pth"

IMG_H = 48
IMG_W = 192
MAX_LEN = 25
TOPK = 8

MAX_EDIT_COST = 4.0
LENGTH_PENALTY = 0.25
EDIT_SCORE_WEIGHT = 1.2
MIN_PROB = 1e-8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
])

CONFUSION_PAIRS = {
    ("O", "0"), ("0", "O"),
    ("I", "1"), ("1", "I"),
    ("I", "L"), ("L", "I"),
    ("S", "5"), ("5", "S"),
    ("B", "8"), ("8", "B"),
    ("C", "O"), ("O", "C"),
    ("E", "F"), ("F", "E"),
    ("E", "L"), ("L", "E"),
    ("R", "P"), ("P", "R"),
    ("N", "H"), ("H", "N"),
    ("M", "N"), ("N", "M"),
    ("A", "R"), ("R", "A"),
    ("T", "I"), ("I", "T"),
    ("U", "V"), ("V", "U"),
    ("D", "O"), ("O", "D"),
    ("G", "C"), ("C", "G"),
}

def idx_to_text(idx):
    if idx == EOS_IDX:
        return "<EOS>"
    if idx == SOS_IDX:
        return "<SOS>"
    if idx >= 3:
        return idx_to_char.get(idx, "")
    return ""

def normalize_text(text):
    result = []
    for ch in text.upper():
        if ch.isalnum():
            result.append(ch)
    return "".join(result)

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
            label = label.replace('"', "").upper()

            data[name] = label

    return data

def build_lexicon(gt_data):
    words = set()

    for label in gt_data.values():
        word = normalize_text(label)

        if 2 <= len(word) <= MAX_LEN:
            words.add(word)

    return sorted(words)

def char_cost(a, b):
    if a == b:
        return 0.0

    if (a, b) in CONFUSION_PAIRS:
        return 0.35

    if a.isdigit() and b.isdigit():
        return 0.6

    if a.isalpha() and b.isalpha():
        return 1.0

    return 1.3

def confusion_edit_distance(a, b):
    n = len(a)
    m = len(b)

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i * 0.9

    for j in range(m + 1):
        dp[0][j] = j * 0.9

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = dp[i - 1][j - 1] + char_cost(a[i - 1], b[j - 1])
            delete = dp[i - 1][j] + 0.9
            insert = dp[i][j - 1] + 0.9

            dp[i][j] = min(sub, delete, insert)

    return dp[n][m]

@torch.no_grad()
def predict_topk_probs(model, image_path):
    img = Image.open(image_path).convert("L")
    x = transform(img).unsqueeze(0).to(DEVICE)

    encoder_outputs, hidden = model.encode(x)

    input_token = torch.tensor(
        [SOS_IDX],
        dtype=torch.long,
        device=DEVICE
    )

    pred_chars = []
    step_probs = []

    for _ in range(MAX_LEN):
        logits, hidden, _ = model.decoder.forward_step(
            input_token,
            hidden,
            encoder_outputs
        )

        probs = F.softmax(logits, dim=-1)

        top_probs, top_indices = torch.topk(
            probs,
            k=TOPK,
            dim=-1
        )

        prob_dict = {}

        for p, idx in zip(top_probs[0], top_indices[0]):
            char = idx_to_text(idx.item())

            if char not in ["", "<SOS>", "<EOS>"]:
                prob_dict[char] = float(p.item())

        best_idx = top_indices[0][0].item()
        best_char = idx_to_text(best_idx)

        if best_idx == EOS_IDX:
            break

        if best_char not in ["", "<SOS>", "<EOS>"]:
            pred_chars.append(best_char)
            step_probs.append(prob_dict)

        input_token = torch.tensor(
            [best_idx],
            dtype=torch.long,
            device=DEVICE
        )

    return "".join(pred_chars), step_probs

def visual_score_candidate(word, step_probs):
    if len(step_probs) == 0:
        return -1e9

    n = len(word)
    m = len(step_probs)

    dp = [[-1e9 for _ in range(m + 1)] for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(n + 1):
        for j in range(m + 1):
            cur = dp[i][j]

            if cur <= -1e8:
                continue

            if i < n and j < m:
                ch = word[i]
                p = step_probs[j].get(ch, MIN_PROB)

                dp[i + 1][j + 1] = max(
                    dp[i + 1][j + 1],
                    cur + math.log(p)
                )

            if j < m:
                dp[i][j + 1] = max(
                    dp[i][j + 1],
                    cur + math.log(0.08)
                )

            if i < n:
                dp[i + 1][j] = max(
                    dp[i + 1][j],
                    cur + math.log(0.05)
                )

    score = dp[n][m]

    score -= LENGTH_PENALTY * abs(n - m)

    return score

def rerank(pred, step_probs, lexicon):
    pred_norm = normalize_text(pred)

    if len(pred_norm) == 0:
        return pred_norm, []

    candidates = []

    pred_len = len(pred_norm)

    for word in lexicon:
        if abs(len(word) - pred_len) > 4:
            continue

        edit_cost = confusion_edit_distance(word, pred_norm)

        if edit_cost > MAX_EDIT_COST:
            continue

        visual_score = visual_score_candidate(word, step_probs)

        final_score = visual_score - EDIT_SCORE_WEIGHT * edit_cost

        candidates.append(
            (
                word,
                final_score,
                visual_score,
                edit_cost
            )
        )

    if len(candidates) == 0:
        return pred_norm, []

    candidates.sort(key=lambda x: x[1], reverse=True)

    return candidates[0][0], candidates[:10]

def main():
    print("device:", DEVICE)
    print("level:", LEVEL)
    print("weights:", WEIGHTS)
    print("max edit cost:", MAX_EDIT_COST)

    gt_data = load_gt(GT_TXT)
    lexicon = build_lexicon(gt_data)

    print("lexicon size:", len(lexicon))

    img_dir = os.path.join(ROOT, LEVEL)

    model = AttentionOCR(vocab_size=VOCAB_SIZE).to(DEVICE)

    model.load_state_dict(
        torch.load(
            WEIGHTS,
            map_location=DEVICE
        )
    )

    model.eval()

    total = 0
    greedy_correct = 0
    rerank_correct = 0
    fallback_count = 0

    wrong_cases = []

    image_files = sorted(os.listdir(img_dir))

    for filename in image_files:
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        gt_name = filename.replace("img_", "word_")
        gt_name = os.path.splitext(gt_name)[0] + ".png"

        if gt_name not in gt_data:
            continue

        gt = normalize_text(gt_data[gt_name])

        if not gt:
            continue

        image_path = os.path.join(img_dir, filename)

        pred, step_probs = predict_topk_probs(
            model,
            image_path
        )

        pred_norm = normalize_text(pred)

        reranked, top_candidates = rerank(
            pred_norm,
            step_probs,
            lexicon
        )

        if len(top_candidates) == 0:
            fallback_count += 1
            reranked = pred_norm

        total += 1

        if pred_norm == gt:
            greedy_correct += 1

        if reranked == gt:
            rerank_correct += 1

        else:
            if len(wrong_cases) < 30:
                wrong_cases.append(
                    (
                        filename,
                        gt,
                        pred_norm,
                        reranked,
                        top_candidates[:5]
                    )
                )

        if total % 200 == 0:
            print(
                f"[{total}] "
                f"greedy={greedy_correct / total:.4f} "
                f"rerank={rerank_correct / total:.4f} "
                f"fallback={fallback_count}"
            )

    print("=" * 60)
    print("level:", LEVEL)
    print("total:", total)
    print("greedy correct:", greedy_correct)
    print("rerank correct:", rerank_correct)
    print("fallback:", fallback_count)
    print("greedy acc:", greedy_correct / total)
    print("rerank acc:", rerank_correct / total)
    print("gain:", rerank_correct / total - greedy_correct / total)
    print("=" * 60)

    print("\nWrong examples:\n")

    for filename, gt, pred, reranked, topc in wrong_cases:
        print(
            f"{filename} | "
            f"GT={gt} | "
            f"PRED={pred} | "
            f"RERANK={reranked}"
        )

        for word, final_score, visual_score, edit_cost in topc:
            print(
                f"   {word:20s} "
                f"final={final_score:.4f} "
                f"visual={visual_score:.4f} "
                f"edit={edit_cost:.2f}"
            )

if __name__ == "__main__":
    main()
