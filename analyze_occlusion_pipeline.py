import os
import difflib
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

IMAGE_PATH = r"D:\mnist_project\ocr1\recognition\recognition\30\img_100.jpg"
GT_TEXT = "FAIRPRICE"

WEIGHTS = r"D:\mnist_project\ocr1\best_attention_istd_occlusion_acc.pth"

OUTPUT_TXT = "occlusion_pipeline_result.txt"

IMG_H = 48
IMG_W = 192
MAX_LEN = 25

TOPK = 5
FIX_THRESHOLD = 0.80
MAX_CANDIDATES = 30

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
        top_char, top_prob, _ = step[0]

        if top_prob >= FIX_THRESHOLD:
            parts.append(top_char)
        else:
            chars = [char for char, _, _ in step]
            parts.append("[" + "/".join(chars) + "]")
            uncertain_positions.append(i)

    return "".join(parts), uncertain_positions

def detect_missing_and_replace(gt, pred):
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

def generate_candidates(topk_steps):
    beams = [("", 1.0)]

    for step in topk_steps:
        new_beams = []

        top_char, top_prob, _ = step[0]

        if top_prob >= FIX_THRESHOLD:
            candidates = [(top_char, top_prob)]
        else:
            candidates = [(char, prob) for char, prob, _ in step]

        for prefix, score in beams:
            for char, prob in candidates:
                new_beams.append(
                    (
                        prefix + char,
                        score * prob
                    )
                )

        new_beams.sort(
            key=lambda x: x[1],
            reverse=True
        )

        beams = new_beams[:MAX_CANDIDATES]

    return beams

def main():
    print("device:", DEVICE)

    model = AttentionOCR(vocab_size=VOCAB_SIZE).to(DEVICE)

    model.load_state_dict(
        torch.load(
            WEIGHTS,
            map_location=DEVICE
        )
    )

    model.eval()

    gt = GT_TEXT.strip().upper()

    pred, topk_steps = predict_topk(
        model,
        IMAGE_PATH
    )

    uncertain_expr, uncertain_positions = build_uncertain_expression(
        topk_steps
    )

    errors = detect_missing_and_replace(
        gt,
        pred
    )

    candidates = generate_candidates(
        topk_steps
    )

    lines = []

    lines.append("=" * 70)
    lines.append("Occlusion Character Analysis Pipeline")
    lines.append("=" * 70)
    lines.append(f"Image:   {IMAGE_PATH}")
    lines.append(f"Weights: {WEIGHTS}")
    lines.append("")
    lines.append(f"GT:      {gt}")
    lines.append(f"Pred:    {pred}")
    lines.append(f"Pattern: {uncertain_expr}")
    lines.append("")

    lines.append("=" * 70)
    lines.append("1. Error Type Analysis")
    lines.append("=" * 70)

    if len(errors) == 0:
        lines.append("No error detected.")
    else:
        for e in errors:
            lines.append(
                f"type={e['type']} | "
                f"gt_pos={e['gt_pos']} | "
                f"gt='{e['gt_text']}' | "
                f"pred_pos={e['pred_pos']} | "
                f"pred='{e['pred_text']}'"
            )

    lines.append("")

    lines.append("=" * 70)
    lines.append("2. Uncertain Positions and Top-k Probability")
    lines.append("=" * 70)

    if len(uncertain_positions) == 0:
        lines.append("No uncertain position.")
    else:
        for pos in uncertain_positions:
            lines.append(f"Position {pos + 1}")
            for char, prob, _ in topk_steps[pos]:
                lines.append(f"  {char:5s} {prob:.4f}")
            lines.append("")

    lines.append("=" * 70)
    lines.append("3. Reconstruction Candidates")
    lines.append("=" * 70)

    for rank, (text, score) in enumerate(candidates, start=1):
        lines.append(
            f"{rank:02d}. {text:20s} score={score:.8f}"
        )

    lines.append("")
    lines.append("=" * 70)
    lines.append("4. Interpretation")
    lines.append("=" * 70)

    lines.append(
        "The model fixes high-confidence characters and gives multiple "
        "candidates for low-confidence positions."
    )
    lines.append(
        "If characters are missing from the prediction, the result indicates "
        "attention alignment compression caused by occlusion."
    )
    lines.append(
        "This output can be used for shape-based character completion without "
        "using a dictionary as the first priority."
    )

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("saved:", OUTPUT_TXT)

if __name__ == "__main__":
    main()
