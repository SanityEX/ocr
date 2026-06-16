import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from torchvision import transforms

from model_attention_ocr import AttentionOCR
from utils import (
    VOCAB_SIZE,
    SOS_IDX,
    EOS_IDX,
    decode_attention
)

IMAGE_PATH = r"D:\mnist_project\ocr1\recognition\recognition\30\img_100.jpg"

WEIGHTS = r"D:\mnist_project\ocr1\best_attention_istd_occlusion_acc.pth"

SAVE_DIR = "attention_vis"

IMG_H = 48
IMG_W = 192
MAX_LEN = 25

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

os.makedirs(SAVE_DIR, exist_ok=True)

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

    return decode_attention(result)

@torch.no_grad()
def predict_with_attention(model, image_path):

    img = Image.open(image_path).convert("L")

    x = transform(img).unsqueeze(0).to(DEVICE)

    encoder_outputs, hidden = model.encode(x)

    input_token = torch.tensor(
        [SOS_IDX],
        dtype=torch.long,
        device=DEVICE
    )

    pred_tokens = []
    attention_maps = []

    for step in range(MAX_LEN):

        logits, hidden, attn = model.decoder.forward_step(
            input_token,
            hidden,
            encoder_outputs
        )

        pred = logits.argmax(-1)

        pred_idx = pred.item()

        if pred_idx == EOS_IDX:
            break

        pred_tokens.append(pred_idx)

        attn = attn.squeeze(0).squeeze(0).cpu()

        attention_maps.append(attn)

        input_token = pred

    pred_text = clean_decode(pred_tokens)

    return img, pred_text, attention_maps

def visualize(img, pred_text, attention_maps):

    img_w, img_h = img.size

    fig, ax = plt.subplots(
        figsize=(16, 4)
    )

    ax.imshow(img, cmap="gray")

    ax.set_title(
        f"Prediction: {pred_text}",
        fontsize=18
    )

    ax.axis("off")

    num_steps = len(attention_maps)

    for i, attn in enumerate(attention_maps):

        pos = torch.argmax(attn).item()

        x_center = (
            pos / len(attention_maps[0])
        ) * img_w

        rect = patches.Rectangle(
            (x_center - 8, 0),
            16,
            img_h,
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )

        ax.add_patch(rect)

        ax.text(
            x_center,
            -5,
            pred_text[i],
            color="red",
            fontsize=14,
            ha="center"
        )

    save_path = os.path.join(
        SAVE_DIR,
        "attention_result.png"
    )

    plt.savefig(
        save_path,
        bbox_inches="tight"
    )

    plt.close()

    print("=" * 50)
    print("Prediction:", pred_text)
    print("Saved:", save_path)
    print("=" * 50)

def main():

    print("device:", DEVICE)

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

    img, pred_text, attention_maps = predict_with_attention(
        model,
        IMAGE_PATH
    )

    visualize(
        img,
        pred_text,
        attention_maps
    )

if __name__ == "__main__":
    main()
