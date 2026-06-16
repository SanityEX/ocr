import os
import random
import shutil
import csv
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np

OUTPUT_DIR = r"D:\mnist_project\ocr1\word_occluded_10k"
FONTS_DIR = r"D:\mnist_project\ocr1\fonts"
COUNT = 10000

IMG_H = 48
IMG_W = 192
CANVAS_W = 320
CANVAS_H = 96

FONT_SIZES = [28, 30, 32, 34, 36, 38]

WORDS = [
    "about", "after", "again", "airport", "avenue", "bank", "beach",
    "black", "blue", "book", "cafe", "center", "city", "coffee",
    "company", "digital", "drive", "family", "food", "garden",
    "green", "hotel", "house", "image", "library", "market",
    "model", "music", "office", "orange", "parking", "phone",
    "picture", "private", "project", "quality", "reader", "review",
    "school", "screen", "service", "station", "street", "system",
    "table", "train", "travel", "window", "winter", "world",
    "yellow", "central", "classic", "network", "monitor",
    "available", "universal", "academy", "journey", "capital"
]

OCCLUSION_TYPES = ["partial", "partial", "partial", "full"]
PARTIAL_DIRECTIONS = ["left", "right", "top", "bottom", "middle"]
PARTIAL_RATIOS = [0.35, 0.5, 0.65]

def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_fonts(fonts_dir):
    fonts = []

    if not os.path.exists(fonts_dir):
        return fonts

    for file in os.listdir(fonts_dir):
        low = file.lower()
        if low.endswith(".ttf") or low.endswith(".otf"):
            fonts.append(os.path.join(fonts_dir, file))

    return fonts

def get_font(font_paths):
    if font_paths:
        return ImageFont.truetype(
            random.choice(font_paths),
            random.choice(FONT_SIZES)
        )

    return ImageFont.load_default()

def random_case(word):
    p = random.random()

    if p < 0.55:
        return word.upper()
    elif p < 0.8:
        return word.lower()
    else:
        return word.capitalize()

def generate_background():
    base = random.randint(215, 255)
    arr = np.ones((CANVAS_H, CANVAS_W), dtype=np.uint8) * base

    noise = np.random.normal(0, random.uniform(1.5, 5.0), arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(arr, mode="L")

def draw_text_with_char_boxes(text, font):
    bg = generate_background()
    draw = ImageDraw.Draw(bg)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = random.randint(10, max(10, CANVAS_W - text_w - 10))
    y = random.randint(20, max(20, CANVAS_H - text_h - 20))

    char_boxes = []

    for ch in text:
        ch_bbox = draw.textbbox((0, 0), ch, font=font)
        ch_w = ch_bbox[2] - ch_bbox[0]
        ch_h = ch_bbox[3] - ch_bbox[1]

        dx = random.randint(-1, 1)
        dy = random.randint(-1, 1)

        px = x + dx
        py = y + dy

        draw.text((px, py), ch, font=font, fill=random.randint(0, 50))

        real_bbox = draw.textbbox((px, py), ch, font=font)

        char_boxes.append({
            "char": ch,
            "box": real_bbox
        })

        x += ch_w + random.randint(0, 2)

    return bg, char_boxes

def apply_real_occlusion(img, char_boxes):
    valid_boxes = [
        item for item in char_boxes
        if item["char"].isalnum()
    ]

    target = random.choice(valid_boxes)

    x1, y1, x2, y2 = target["box"]

    pad = 1
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img.size[0], x2 + pad)
    y2 = min(img.size[1], y2 + pad)

    occ_type = random.choice(OCCLUSION_TYPES)
    direction = "none"
    ratio = 1.0

    w = x2 - x1
    h = y2 - y1

    if occ_type == "full":
        erase_box = (x1, y1, x2, y2)
    else:
        direction = random.choice(PARTIAL_DIRECTIONS)
        ratio = random.choice(PARTIAL_RATIOS)

        if direction == "left":
            erase_box = (x1, y1, x1 + int(w * ratio), y2)

        elif direction == "right":
            erase_box = (x2 - int(w * ratio), y1, x2, y2)

        elif direction == "top":
            erase_box = (x1, y1, x2, y1 + int(h * ratio))

        elif direction == "bottom":
            erase_box = (x1, y2 - int(h * ratio), x2, y2)

        else:
            mx1 = x1 + int(w * 0.25)
            mx2 = x2 - int(w * 0.25)
            erase_box = (mx1, y1, mx2, y2)

    draw = ImageDraw.Draw(img)
    draw.rectangle(erase_box, fill=255)

    return img, {
        "occluded_char": target["char"],
        "char_box": (x1, y1, x2, y2),
        "erase_box": erase_box,
        "occlusion_type": occ_type,
        "direction": direction,
        "ratio": ratio
    }

def degrade_image(img):
    if random.random() < 0.35:
        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(0.3, 1.0)
            )
        )

    if random.random() < 0.25:
        img = ImageEnhance.Contrast(img).enhance(
            random.uniform(0.75, 1.25)
        )

    if random.random() < 0.25:
        img = ImageEnhance.Brightness(img).enhance(
            random.uniform(0.85, 1.15)
        )

    return img

def crop_and_resize(img):
    arr = np.array(img)
    mask = arr < 245

    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return img.resize((IMG_W, IMG_H))

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    pad_x = 8
    pad_y = 5

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(arr.shape[1] - 1, x2 + pad_x)
    y2 = min(arr.shape[0] - 1, y2 + pad_y)

    cropped = img.crop((x1, y1, x2 + 1, y2 + 1))
    return cropped.resize((IMG_W, IMG_H), Image.BICUBIC)

def main():
    reset_dir(OUTPUT_DIR)

    img_dir = os.path.join(OUTPUT_DIR, "images")
    ensure_dir(img_dir)

    labels_path = os.path.join(OUTPUT_DIR, "labels.txt")
    meta_path = os.path.join(OUTPUT_DIR, "metadata.csv")

    font_paths = load_fonts(FONTS_DIR)

    label_lines = []

    with open(meta_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "filename",
            "label",
            "occluded_char",
            "occlusion_type",
            "direction",
            "ratio",
            "char_box",
            "erase_box"
        ])

        for i in range(COUNT):
            word = random_case(random.choice(WORDS))
            font = get_font(font_paths)

            img, char_boxes = draw_text_with_char_boxes(word, font)
            img, info = apply_real_occlusion(img, char_boxes)
            img = degrade_image(img)
            img = crop_and_resize(img)

            filename = f"word_occ_{i:06d}.png"
            save_path = os.path.join(img_dir, filename)

            img.save(save_path)

            label_lines.append(f"{filename}\t{word}")

            writer.writerow([
                filename,
                word,
                info["occluded_char"],
                info["occlusion_type"],
                info["direction"],
                info["ratio"],
                info["char_box"],
                info["erase_box"]
            ])

            if (i + 1) % 500 == 0:
                print(f"generated: {i + 1}/{COUNT}")

    with open(labels_path, "w", encoding="utf-8") as f:
        f.write("\n".join(label_lines))

    print("=" * 50)
    print("done")
    print("output:", OUTPUT_DIR)
    print("images:", img_dir)
    print("labels:", labels_path)
    print("metadata:", meta_path)
    print("=" * 50)

if __name__ == "__main__":
    main()
