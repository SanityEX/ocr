import os
import difflib
import csv
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model_attention_ocr import AttentionOCR
from utils import (
    VOCAB_SIZE,
    SOS_IDX,
    EOS_IDX,
    idx_to_char,
    decode_attention
)


ROOT = r"D:\mnist_project\ocr1\recognition\recognition"
LEVEL = "30"
GT_TXT = r"D:\mnist_project\ocr1\recognition\recognition\gt_recognition.txt"
WEIGHTS = r"D:\mnist_project\ocr1\best_attention_istd_occlusion_acc.pth"

OUTPUT_TXT = "occlusion_pipeline_batch_result.txt"
OUTPUT_CSV = "occlusion_pipeline_batch_result.csv"

IMG_H = 48
IMG_W = 192
MAX_LEN = 25
TOPK = 5
FIX_THRESHOLD = 0.80
MAX_SAMPLES = 1000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
])


def idx_to_text(idx):
    if idx == EOS_IDX:
        return "<EOS>"
    if idx == SOS_IDX:
        return "<SOS>"
    if idx >= 3:
        return idx_to_char.get(idx, "")
    return ""


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


def clean_decode(seq):
    result = []

    for idx in seq:
        if idx == EOS_IDX:
            break

        if idx >= 3:
            result.append(idx)

    return decode_attention(result).strip().upper()


@torch.no_grad()
def predict_topk(model, image_path):
    img = Image.open(image_path).convert("L")
    x = transform(img).unsqueeze(0).to(DEVICE)

    encoder_outputs, hidden = model.encode(x)

    input_token = torch.tensor(
        [SOS_IDX],
        dtype=torch.long,
        device=DEVICE
    )

    pred_tokens = []
    topk_steps = []

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

        step = []

        for prob, idx in zip(top_probs[0], top_indices[0]):
            idx = idx.item()
            char = idx_to_text(idx)

            if char not in ["", "<SOS>", "<EOS>"]:
                step.append((char, prob.item(), idx))

        best_idx = top_indices[0][0].item()

        if best_idx == EOS_IDX:
            break

        pred_tokens.append(best_idx)
        topk_steps.append(step)

        input_token = torch.tensor(
            [best_idx],
            dtype=torch.long,
            device=DEVICE
        )

    pred_text = clean_decode(pred_tokens)

    return pred_text, topk_steps


def build_uncertain_expression(topk_steps):
    parts = []
    uncertain_positions = []

    for i, step in enumerate(topk_steps):
        if len(step) == 0:
            continue

        top_char, top_prob, _ = step[0]

        if top_prob >= FIX_THRESHOLD:
            parts.append(top_char)
        else:
            chars = [char for char, _, _ in step]
            parts.append("[" + "/".join(chars) + "]")
            uncertain_positions.append(i)

    return "".join(parts), uncertain_positions


def detect_errors(gt, pred):
    matcher = difflib.SequenceMatcher(None, gt, pred)
    results = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "delete":
            results.append({
                "type": "missing",
                "gt_pos": i1,
                "gt_text": gt[i1:i2],
                "pred_pos": j1,
                "pred_text": ""
            })

        elif tag == "replace":
            results.append({
                "type": "replace",
                "gt_pos": i1,
                "gt_text": gt[i1:i2],
                "pred_pos": j1,
                "pred_text": pred[j1:j2]
            })

        elif tag == "insert":
            results.append({
                "type": "extra",
                "gt_pos": i1,
                "gt_text": "",
                "pred_pos": j1,
                "pred_text": pred[j1:j2]
            })

    return results


def generate_candidates(topk_steps, max_candidates=10):
    beams = [("", 1.0)]

    for step in topk_steps:
        if len(step) == 0:
            continue

        top_char, top_prob, _ = step[0]

        if top_prob >= FIX_THRESHOLD:
            candidates = [(top_char, top_prob)]
        else:
            candidates = [(char, prob) for char, prob, _ in step]

        new_beams = []

        for prefix, score in beams:
            for char, prob in candidates:
                new_beams.append((prefix + char, score * prob))

        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:max_candidates]

    return beams


def format_topk_for_position(topk_steps, pos):
    if pos < 0 or pos >= len(topk_steps):
        return ""

    items = []

    for char, prob, _ in topk_steps[pos]:
        items.append(f"{char}:{prob:.4f}")

    return " / ".join(items)


def main():
    print("device:", DEVICE)
    print("level:", LEVEL)

    img_dir = os.path.join(ROOT, LEVEL)
    gt_data = load_gt(GT_TXT)

    model = AttentionOCR(vocab_size=VOCAB_SIZE).to(DEVICE)

    model.load_state_dict(
        torch.load(
            WEIGHTS,
            map_location=DEVICE
        )
    )

    model.eval()

    image_files = sorted(os.listdir(img_dir))

    text_lines = []
    csv_rows = []

    total = 0
    correct = 0

    replace_count = 0
    missing_count = 0
    extra_count = 0

    top5_hit_in_error = 0
    error_char_count = 0

    for filename in image_files:
        if total >= MAX_SAMPLES:
            break

        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        gt_name = filename.replace("img_", "word_")
        gt_name = os.path.splitext(gt_name)[0] + ".png"

        if gt_name not in gt_data:
            continue

        image_path = os.path.join(img_dir, filename)
        gt = gt_data[gt_name].upper()

        pred, topk_steps = predict_topk(model, image_path)

        errors = detect_errors(gt, pred)
        pattern, uncertain_positions = build_uncertain_expression(topk_steps)
        candidates = generate_candidates(topk_steps, max_candidates=10)

        ok = gt == pred

        total += 1

        if ok:
            correct += 1

        for e in errors:
            if e["type"] == "replace":
                replace_count += 1
            elif e["type"] == "missing":
                missing_count += 1
            elif e["type"] == "extra":
                extra_count += 1

            gt_pos = e["gt_pos"]

            if e["gt_text"]:
                for offset, ch in enumerate(e["gt_text"]):
                    pred_pos = gt_pos + offset

                    if pred_pos < len(topk_steps):
                        topk_chars = [x[0] for x in topk_steps[pred_pos]]
                        error_char_count += 1

                        if ch in topk_chars:
                            top5_hit_in_error += 1

        text_lines.append("=" * 80)
        text_lines.append(f"file: {filename}")
        text_lines.append(f"GT:      {gt}")
        text_lines.append(f"Pred:    {pred}")
        text_lines.append(f"Pattern: {pattern}")
        text_lines.append("")

        if len(errors) == 0:
            text_lines.append("Errors: none")
        else:
            text_lines.append("Errors:")
            for e in errors:
                text_lines.append(
                    f"  type={e['type']} | "
                    f"gt_pos={e['gt_pos']} | "
                    f"gt='{e['gt_text']}' | "
                    f"pred_pos={e['pred_pos']} | "
                    f"pred='{e['pred_text']}'"
                )

        text_lines.append("")

        if len(uncertain_positions) > 0:
            text_lines.append("Uncertain positions:")
            for pos in uncertain_positions:
                text_lines.append(
                    f"  pos {pos + 1}: {format_topk_for_position(topk_steps, pos)}"
                )
        else:
            text_lines.append("Uncertain positions: none")

        text_lines.append("")

        text_lines.append("Top reconstruction candidates:")
        for rank, (cand, score) in enumerate(candidates, start=1):
            text_lines.append(f"  {rank:02d}. {cand:20s} score={score:.8f}")

        text_lines.append("")

        csv_rows.append({
            "filename": filename,
            "gt": gt,
            "pred": pred,
            "correct": int(ok),
            "pattern": pattern,
            "errors": str(errors),
            "top_candidate": candidates[0][0] if candidates else "",
            "top_candidate_score": candidates[0][1] if candidates else 0.0
        })

        if total % 50 == 0:
            print(f"[{total}] acc={correct / total:.4f}")

    acc = correct / total if total > 0 else 0.0
    top5_error_hit_rate = (
        top5_hit_in_error / error_char_count
        if error_char_count > 0
        else 0.0
    )

    summary = []
    summary.append("=" * 80)
    summary.append("Occlusion Pipeline Batch Summary")
    summary.append("=" * 80)
    summary.append(f"Level: {LEVEL}")
    summary.append(f"Total samples: {total}")
    summary.append(f"Correct: {correct}")
    summary.append(f"Accuracy: {acc:.4f}")
    summary.append(f"Replace errors: {replace_count}")
    summary.append(f"Missing errors: {missing_count}")
    summary.append(f"Extra errors: {extra_count}")
    summary.append(f"Top-5 hit rate in error positions: {top5_error_hit_rate:.4f}")
    summary.append("=" * 80)
    summary.append("")

    text_lines = summary + text_lines

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(text_lines))

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "filename",
            "gt",
            "pred",
            "correct",
            "pattern",
            "errors",
            "top_candidate",
            "top_candidate_score"
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in csv_rows:
            writer.writerow(row)

    print("\n".join(summary))
    print("saved:", OUTPUT_TXT)
    print("saved:", OUTPUT_CSV)


if __name__ == "__main__":
    main()