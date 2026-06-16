import os
import random
import shutil
import string
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUTPUT_DIR = "data_realprint"
TRAIN_COUNT = 5000
VAL_COUNT = 800
TEST_COUNT = 800

IMG_W = 140
IMG_H = 52

FONTS_DIR = "fonts"
FONT_SIZES = [24, 26, 28, 30, 32]

COMMON_WORDS = [
    "hello", "python", "world", "model", "train", "data", "batch", "image",
    "tensor", "label", "epoch", "loss", "code", "debug", "vision", "text",
    "random", "rotate", "predict", "reader", "system", "neural", "sample",
    "test", "value", "input", "output", "layer", "learn", "network",
    "feature", "vector", "result", "object", "window", "orange",
    "school", "paper", "button", "screen", "editor", "camera", "filter"
]

def reset_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_fonts(fonts_dir: str):
    fonts = []
    if not os.path.exists(fonts_dir):
        print(f"[WARN] fonts dir not found: {fonts_dir}")
        return fonts

    for file in os.listdir(fonts_dir):
        name = file.lower()
        if not name.endswith((".ttf", ".otf")):
            continue
        if "italic" in name or "oblique" in name:
            continue
        if name.endswith("i.ttf") or name.endswith("z.ttf"):
            continue
        fonts.append(os.path.join(fonts_dir, file))
    return fonts

def random_word():
    if random.random() < 0.8:
        return random.choice(COMMON_WORDS)
    length = random.randint(3, 8)
    return "".join(random.choices(string.ascii_lowercase, k=length))

def add_noise(img: Image.Image, sigma=4):
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def render_text_image(text: str, font_paths: list[str]):
    bg_color = random.randint(225, 255)
    fg_color = random.randint(0, 35)

    img = Image.new("L", (IMG_W, IMG_H), color=bg_color)
    draw = ImageDraw.Draw(img)

    if font_paths:
        font_path = random.choice(font_paths)
        font_size = random.choice(FONT_SIZES)
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = max(4, (IMG_W - text_w) // 2 + random.randint(-10, 10))
    y = max(2, (IMG_H - text_h) // 2 + random.randint(-4, 4))

    draw.text((x, y), text, font=font, fill=fg_color)

    angle = random.uniform(-3, 3)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=bg_color)

    if random.random() < 0.2:
        new_w = random.randint(IMG_W - 8, IMG_W)
        new_h = random.randint(IMG_H - 4, IMG_H)
        img = img.resize((new_w, new_h), Image.BICUBIC)
        canvas = Image.new("L", (IMG_W, IMG_H), color=bg_color)
        paste_x = random.randint(0, IMG_W - new_w)
        paste_y = random.randint(0, IMG_H - new_h)
        canvas.paste(img, (paste_x, paste_y))
        img = canvas

    if random.random() < 0.15:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))

    if random.random() < 0.2:
        img = add_noise(img, sigma=random.uniform(2, 6))

    return img

def save_split(split_name: str, count: int, font_paths: list[str], start_idx: int):
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    image_dir = os.path.join(split_dir, "images")
    ensure_dir(image_dir)

    label_path = os.path.join(split_dir, "labels.txt")
    lines = []

    for i in range(count):
        text = random_word()
        img = render_text_image(text, font_paths)

        file_id = start_idx + i
        filename = f"image_{file_id:05d}.png"
        img.save(os.path.join(image_dir, filename))
        lines.append(f"{filename} {text}")

    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"{split_name}: {count} images saved.")

def main():
    reset_dir(OUTPUT_DIR)
    font_paths = load_fonts(FONTS_DIR)
    print(f"fonts loaded: {len(font_paths)}")

    save_split("train", TRAIN_COUNT, font_paths, 0)
    save_split("val", VAL_COUNT, font_paths, TRAIN_COUNT)
    save_split("test", TEST_COUNT, font_paths, TRAIN_COUNT + VAL_COUNT)

    print("real print dataset generation finished.")

if __name__ == "__main__":
    main()
