import os
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
    idx_to_char
)

IMG_DIR = r"D:\mnist_project\ocr1\word_occluded_10k\images"
META_CSV = r"D:\mnist_project\ocr1\word_occluded_10k\metadata.csv"
WEIGHTS = r"D:\mnist_project\ocr1\best_attention_v2_phase2_acc.pth"
OUTPUT_TXT = "occluded_topk_result.txt"

IMG_H = 48
IMG_W = 192
MAX_LEN = 25
TOPK = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
])

def load_meta(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def idx_to_text(idx):
    if idx == EOS_IDX:
        return "<EOS>"
    if idx == SOS_IDX:
        return "<SOS>"
    if idx >= 3:
        return idx_to_char.get(idx, "")
    return ""

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

    pred_chars = []
    topk_steps = []

    for step in range(MAX_LEN):
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

        step_candidates = []

        for p, idx in zip(top_probs[0], top_indices[0]):
            char = idx_to_text(idx.item())
            prob = p.item()
            step_candidates.append((char, prob))

        best_idx = top_indices[0][0].item()

        if best_idx == EOS_IDX:
            break

        best_char = idx_to_text(best_idx)
        pred_chars.append(best_char)

        topk_steps.append(step_candidates)

        input_token = torch.tensor(
            [best_idx],
            dtype=torch.long,
            device=DEVICE
        )

    return "".join(pred_chars), topk_steps

def main():
    print("device:", DEVICE)

    model = AttentionOCR(vocab_size=VOCAB_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
    model.eval()

    rows = load_meta(META_CSV)

    results = []

    for i, row in enumerate(rows[:50]):
        filename = row["filename"]
        label = row["label"]
        occ_char = row["occluded_char"]
        char_index = int(row.get("char_index", -1)) if "char_index" in row else -1
        occ_type = row["occlusion_type"]
        direction = row["direction"]
        ratio = row["ratio"]

        img_path = os.path.join(IMG_DIR, filename)

        if not os.path.exists(img_path):
            continue

        pred, topk_steps = predict_topk(model, img_path)

        results.append("=" * 70)
        results.append(f"file: {filename}")
        results.append(f"GT:   {label}")
        results.append(f"Pred: {pred}")
        results.append(f"occluded_char: {occ_char}")
        results.append(f"type={occ_type}, direction={direction}, ratio={ratio}")

        results.append("")

        if 0 <= char_index < len(topk_steps):
            results.append(f"Top-{TOPK} at occluded position {char_index}:")
            for char, prob in topk_steps[char_index]:
                results.append(f"  {char:5s} : {prob:.4f}")
        else:
            results.append("occluded position not found in prediction length")

        results.append("")

        if (i + 1) % 10 == 0:
            print(f"processed: {i + 1}")

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(results))

    print("saved:", OUTPUT_TXT)

if __name__ == "__main__":
    main()
