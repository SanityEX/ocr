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
    char_to_idx
)

ROOT = r"D:\mnist_project\ocr1\recognition\recognition"

LEVEL = "30"

GT_TXT = r"D:\mnist_project\ocr1\recognition\recognition\gt_recognition.txt"

WEIGHTS = r"D:\mnist_project\ocr1\best_attention_istd_occlusion_acc.pth"

OUTPUT_TXT = "topk_batch_analysis.txt"

TOPK = 5
MAX_SAMPLES = 1000

IMG_H = 48
IMG_W = 192
MAX_LEN = 25

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
])

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
            label = label.upper()

            data[name] = label

    return data

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

    topk_result = []

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

        step_candidates = []

        for p, idx in zip(top_probs[0], top_indices[0]):
            step_candidates.append(
                (
                    idx.item(),
                    p.item()
                )
            )

        best_idx = top_indices[0][0].item()

        if best_idx == EOS_IDX:
            break

        topk_result.append(step_candidates)

        input_token = torch.tensor(
            [best_idx],
            dtype=torch.long,
            device=DEVICE
        )

    return topk_result

def main():

    print("device:", DEVICE)
    print("level:", LEVEL)

    gt_data = load_gt(GT_TXT)

    img_dir = os.path.join(ROOT, LEVEL)

    model = AttentionOCR(
        vocab_size=VOCAB_SIZE
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            WEIGHTS,
            map_location=DEVICE
        )
    )

    model.eval()

    total_chars = 0

    top1_correct = 0
    top3_correct = 0
    top5_correct = 0

    interesting_cases = []

    image_files = sorted(os.listdir(img_dir))

    processed = 0

    for filename in image_files:

        if processed >= MAX_SAMPLES:
            break

        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        gt_name = filename.replace("img_", "word_")
        gt_name = os.path.splitext(gt_name)[0] + ".png"

        if gt_name not in gt_data:
            continue

        gt_text = gt_data[gt_name]

        image_path = os.path.join(
            img_dir,
            filename
        )

        topk_result = predict_topk(
            model,
            image_path
        )

        min_len = min(
            len(gt_text),
            len(topk_result)
        )

        for pos in range(min_len):

            gt_char = gt_text[pos]

            if gt_char not in char_to_idx:
                continue

            gt_idx = char_to_idx[gt_char]

            candidates = topk_result[pos]

            top1 = [x[0] for x in candidates[:1]]
            top3 = [x[0] for x in candidates[:3]]
            top5 = [x[0] for x in candidates[:5]]

            total_chars += 1

            if gt_idx in top1:
                top1_correct += 1

            if gt_idx in top3:
                top3_correct += 1

            if gt_idx in top5:
                top5_correct += 1

                if gt_idx not in top1:
                    if len(interesting_cases) < 30:

                        interesting_cases.append({
                            "file": filename,
                            "position": pos,
                            "gt_char": gt_char,
                            "topk": candidates
                        })

        processed += 1

        if processed % 100 == 0:
            print(f"processed: {processed}")

    top1_acc = top1_correct / total_chars
    top3_acc = top3_correct / total_chars
    top5_acc = top5_correct / total_chars

    lines = []

    lines.append("=" * 60)
    lines.append(f"LEVEL: {LEVEL}")
    lines.append(f"SAMPLES: {processed}")
    lines.append(f"TOTAL CHARS: {total_chars}")
    lines.append("=" * 60)

    lines.append(f"Top-1 Accuracy : {top1_acc:.4f}")
    lines.append(f"Top-3 Coverage : {top3_acc:.4f}")
    lines.append(f"Top-5 Coverage : {top5_acc:.4f}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("Interesting Cases")
    lines.append("=" * 60)

    for case in interesting_cases:

        lines.append(
            f"{case['file']} "
            f"pos={case['position']} "
            f"GT={case['gt_char']}"
        )

        for idx, prob in case["topk"]:

            char = "?"

            for k, v in char_to_idx.items():
                if v == idx:
                    char = k
                    break

            lines.append(
                f"   {char:5s} {prob:.4f}"
            )

        lines.append("")

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("saved:", OUTPUT_TXT)

if __name__ == "__main__":
    main()
