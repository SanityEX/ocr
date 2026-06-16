import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model_attention_ocr import AttentionOCR
from utils import (
    VOCAB_SIZE,
    SOS_IDX,
    EOS_IDX,
    decode_attention,
    idx_to_char
)

IMAGE_PATH = r"D:\mnist_project\ocr1\recognition\recognition\30\img_100.jpg"
WEIGHTS = r"D:\mnist_project\ocr1\best_attention_istd_occlusion_acc.pth"

OUTPUT_TXT = "topk_probability_result.txt"

IMG_H = 48
IMG_W = 192
MAX_LEN = 25
TOPK = 5

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
    all_topk = []

    for step in range(MAX_LEN):
        logits, hidden, attn = model.decoder.forward_step(
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

        step_topk = []

        for p, idx in zip(top_probs[0], top_indices[0]):
            char = idx_to_text(idx.item())
            prob = p.item()
            step_topk.append((char, prob))

        best_idx = top_indices[0][0].item()

        if best_idx == EOS_IDX:
            break

        pred_tokens.append(best_idx)
        all_topk.append(step_topk)

        input_token = torch.tensor(
            [best_idx],
            dtype=torch.long,
            device=DEVICE
        )

    pred_text = clean_decode(pred_tokens)

    return pred_text, all_topk

def main():
    print("device:", DEVICE)
    print("image:", IMAGE_PATH)
    print("weights:", WEIGHTS)

    model = AttentionOCR(vocab_size=VOCAB_SIZE).to(DEVICE)

    model.load_state_dict(
        torch.load(
            WEIGHTS,
            map_location=DEVICE
        )
    )

    model.eval()

    pred_text, all_topk = predict_topk(model, IMAGE_PATH)

    lines = []

    lines.append("=" * 60)
    lines.append(f"Image: {IMAGE_PATH}")
    lines.append(f"Weights: {WEIGHTS}")
    lines.append(f"Prediction: {pred_text}")
    lines.append("=" * 60)
    lines.append("")

    for i, topk in enumerate(all_topk):
        char = pred_text[i] if i < len(pred_text) else ""

        lines.append(f"Position {i + 1}  Pred: {char}")
        for cand, prob in topk:
            lines.append(f"  {cand:6s} {prob:.4f}")
        lines.append("")

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("saved:", OUTPUT_TXT)

if __name__ == "__main__":
    main()
