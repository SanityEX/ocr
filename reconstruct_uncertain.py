import os
import itertools
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
WEIGHTS = r"D:\mnist_project\ocr1\best_attention_istd_occlusion_acc.pth"

OUTPUT_TXT = "uncertain_reconstruction_result.txt"

IMG_H = 48
IMG_W = 192
MAX_LEN = 25

TOPK = 5
FIX_THRESHOLD = 0.80
LOW_CONF_THRESHOLD = 0.50
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
    return decode_attention(result)

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

def build_uncertain_pattern(topk_steps):
    pattern = []
    uncertain_positions = []

    for i, step in enumerate(topk_steps):
        top_char, top_prob, _ = step[0]

        if top_prob >= FIX_THRESHOLD:
            pattern.append({
                "type": "fixed",
                "chars": [(top_char, top_prob)]
            })
        else:
            candidates = [
                (char, prob)
                for char, prob, _ in step
                if prob >= 0.01
            ]

            pattern.append({
                "type": "uncertain",
                "chars": candidates
            })

            uncertain_positions.append(i)

    return pattern, uncertain_positions

def generate_candidates(pattern):
    beams = [("", 1.0)]

    for item in pattern:
        new_beams = []

        for prefix, score in beams:
            for char, prob in item["chars"]:
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

def make_visual_expression(pattern):
    parts = []

    for item in pattern:
        if item["type"] == "fixed":
            parts.append(item["chars"][0][0])
        else:
            chars = [c for c, _ in item["chars"]]
            parts.append("[" + "/".join(chars) + "]")

    return "".join(parts)

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

    pred_text, topk_steps = predict_topk(
        model,
        IMAGE_PATH
    )

    pattern, uncertain_positions = build_uncertain_pattern(
        topk_steps
    )

    candidates = generate_candidates(pattern)

    visual_expression = make_visual_expression(pattern)

    lines = []

    lines.append("=" * 70)
    lines.append(f"Image: {IMAGE_PATH}")
    lines.append(f"Weights: {WEIGHTS}")
    lines.append(f"Top-1 Prediction: {pred_text}")
    lines.append(f"Uncertain Expression: {visual_expression}")
    lines.append("=" * 70)
    lines.append("")

    lines.append("Uncertain Positions")
    lines.append("-" * 70)

    for pos in uncertain_positions:
        lines.append(f"Position {pos + 1}")
        for char, prob, _ in topk_steps[pos]:
            lines.append(f"  {char:5s} {prob:.4f}")
        lines.append("")

    lines.append("=" * 70)
    lines.append("Reconstruction Candidates")
    lines.append("=" * 70)

    for rank, (text, score) in enumerate(candidates, start=1):
        lines.append(
            f"{rank:02d}. {text:20s} score={score:.8f}"
        )

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("saved:", OUTPUT_TXT)

if __name__ == "__main__":
    main()
