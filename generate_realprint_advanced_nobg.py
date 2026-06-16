import os
import random
import shutil
import string
import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

OUTPUT_DIR = "data_realprint_realistic_100k"
FONTS_DIR = "fonts"

TRAIN_COUNT = 100000
VAL_COUNT = 5000
TEST_COUNT = 5000

IMG_H = 48
MIN_W = 48
MAX_W = 256

CANVAS_W = 320
CANVAS_H = 96

FONT_SIZES = [24, 26, 28, 30, 32, 34, 36, 38]

CHARS = string.ascii_lowercase + string.ascii_uppercase + string.digits

COMMON_WORDS = [
    "hello", "python", "world", "model", "train", "data", "batch", "image",
    "tensor", "label", "epoch", "loss", "code", "debug", "vision", "text",
    "random", "rotate", "predict", "reader", "system", "neural", "sample",
    "test", "value", "input", "output", "layer", "learn", "network",
    "feature", "vector", "result", "object", "window", "orange",
    "school", "paper", "button", "screen", "editor", "camera", "filter",
    "google", "amazon", "microsoft", "OpenAI", "Apple", "NVIDIA",
    "facebook", "youtube", "instagram", "telegram", "twitter", "github",
    "docker", "linux", "ubuntu", "visual", "studio", "project",
    "masterfile", "advertising", "center", "review", "journey",
    "private", "parking", "salutes", "channel", "universal",
    "hangover", "divorce", "blush", "ultimate", "november",
    "speech", "available", "avenue", "hudgens", "monitor", "picture",
    "plastic", "academy", "quality", "airport", "station", "digital",
    "central", "network", "classic", "justice", "library", "capital",
    "hotel", "street", "store", "market", "coffee", "office", "garden",
    "music", "phone", "photo", "global", "service", "travel", "yellow",
    "silver", "bridge", "summer", "winter", "future", "energy"
]

CONFUSION_PATTERNS = [
    "O0O0O", "0O0O0", "I1I1I", "1I1I1", "B8B8", "8B8B",
    "S5S5", "5S5S", "Z2Z2", "2Z2Z", "G6G6", "6G6G",
    "A4A4", "4A4A", "O0I1", "B8S5", "I1l1",
    "rnrn", "mrmr", "vvww", "wvwv", "mnrn", "vvwvv",
    "HUDGENS", "SPEECH", "AVAILABLE", "OUTSLUITLIKE", "AVENUE",
    "STATION", "AIRPORT", "DIGITAL", "CENTRAL", "QUALITY",
    "HOTEL2024", "SHOP88", "OPEN24", "SALE50", "GATE01"
]

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
        font_path = random.choice(font_paths)
        font_size = random.choice(FONT_SIZES)
        return ImageFont.truetype(font_path, font_size)

    return ImageFont.load_default()

def random_case(word):
    mode = random.random()

    if mode < 0.35:
        return word.lower()
    elif mode < 0.55:
        return word.upper()
    elif mode < 0.75:
        return word.capitalize()
    else:
        chars = []
        for c in word:
            if c.isalpha():
                chars.append(c.upper() if random.random() < 0.5 else c.lower())
            else:
                chars.append(c)
        return "".join(chars)

def random_alnum_string(min_len=3, max_len=14):
    length = random.randint(min_len, max_len)
    return "".join(random.choices(CHARS, k=length))

def random_long_word():
    length = random.randint(10, 18)
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))

def random_number_string():
    length = random.randint(3, 10)
    return "".join(random.choices(string.digits, k=length))

def random_word():
    p = random.random()

    if p < 0.20:
        return random_case(random.choice(COMMON_WORDS))

    elif p < 0.35:
        word = random_case(random.choice(COMMON_WORDS))
        if random.random() < 0.5:
            return word + str(random.randint(0, 9999))
        else:
            return str(random.randint(0, 9999)) + word

    elif p < 0.50:
        return random_alnum_string(3, 14)

    elif p < 0.80:
        return random.choice(CONFUSION_PATTERNS)

    elif p < 0.90:
        return random_number_string()

    else:
        return random_long_word()

def generate_clean_background(width, height):
    base = random.randint(205, 255)
    arr = np.ones((height, width), dtype=np.uint8) * base

    if random.random() < 0.8:
        grad = np.linspace(
            random.randint(-25, 25),
            random.randint(-25, 25),
            width
        ).astype(np.int16)
        arr = np.clip(arr.astype(np.int16) + grad, 0, 255).astype(np.uint8)

    if random.random() < 0.8:
        noise = np.random.normal(0, random.uniform(2, 10), (height, width))
        arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(arr, mode="L")

def choose_text_gray(bg_img):
    bg_mean = np.array(bg_img).mean()

    if bg_mean > 220:
        return random.randint(0, 65)
    elif bg_mean > 180:
        return random.randint(20, 95)
    else:
        return random.randint(40, 130)

def create_text_layer(text, font, canvas_size):
    width, height = canvas_size
    layer = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(layer)

    color = random.randint(180, 255)

    bbox = draw.textbbox((0, 0), text, font=font)
    total_w = bbox[2] - bbox[0]
    total_h = bbox[3] - bbox[1]

    start_x = random.randint(5, max(5, width - total_w - 5))
    baseline_y = random.randint(8, max(8, height - total_h - 8))

    x = start_x

    for ch in text:
        ch_bbox = draw.textbbox((0, 0), ch, font=font)
        ch_w = ch_bbox[2] - ch_bbox[0]

        dx = random.randint(-1, 2)
        dy = random.randint(-2, 2)
        y = baseline_y + dy

        draw.text((x + dx, y), ch, font=font, fill=color)

        if random.random() < 0.15:
            draw.text((x + dx + 1, y), ch, font=font, fill=color)

        x += ch_w + random.randint(0, 3)

    return layer

def blend_text(bg_img, text_mask):
    bg = np.array(bg_img).astype(np.float32)
    mask = np.array(text_mask).astype(np.float32) / 255.0

    fg = choose_text_gray(bg_img)
    alpha = random.uniform(0.65, 1.0)

    out = bg * (1 - mask * alpha) + fg * (mask * alpha)
    out = np.clip(out, 0, 255).astype(np.uint8)

    return Image.fromarray(out, mode="L")

def find_perspective_coeffs(src_pts, dst_pts):
    matrix = []

    for p1, p2 in zip(dst_pts, src_pts):
        matrix.append([
            p1[0], p1[1], 1,
            0, 0, 0,
            -p2[0] * p1[0],
            -p2[0] * p1[1]
        ])
        matrix.append([
            0, 0, 0,
            p1[0], p1[1], 1,
            -p2[1] * p1[0],
            -p2[1] * p1[1]
        ])

    A = np.array(matrix, dtype=np.float32)
    B = np.array(src_pts).reshape(8)

    coeffs = np.linalg.lstsq(A, B, rcond=None)[0]
    return coeffs

def random_perspective(img):
    w, h = img.size
    mx = int(w * 0.08)
    my = int(h * 0.15)

    src = [(0, 0), (w, 0), (w, h), (0, h)]

    dst = [
        (random.randint(0, mx), random.randint(0, my)),
        (w - random.randint(0, mx), random.randint(0, my)),
        (w - random.randint(0, mx), h - random.randint(0, my)),
        (random.randint(0, mx), h - random.randint(0, my)),
    ]

    coeffs = find_perspective_coeffs(src, dst)

    return img.transform(
        (w, h),
        Image.PERSPECTIVE,
        coeffs,
        Image.BICUBIC
    )

def random_affine(img):
    angle = random.uniform(-6, 6)

    img = img.rotate(
        angle,
        resample=Image.BICUBIC,
        expand=False,
        fillcolor=255
    )

    if random.random() < 0.5:
        dx = random.uniform(-0.05, 0.05) * img.size[0]
        dy = random.uniform(-0.05, 0.05) * img.size[1]

        img = img.transform(
            img.size,
            Image.AFFINE,
            (1, 0, dx, 0, 1, dy),
            resample=Image.BICUBIC,
            fillcolor=255
        )

    return img

def add_gaussian_noise(img, sigma=8):
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)

    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(arr, mode="L")

def add_salt_pepper_noise(img, amount=0.002):
    arr = np.array(img).copy()
    num = int(amount * arr.size)

    ys = np.random.randint(0, arr.shape[0], num)
    xs = np.random.randint(0, arr.shape[1], num)
    arr[ys, xs] = 255

    ys = np.random.randint(0, arr.shape[0], num)
    xs = np.random.randint(0, arr.shape[1], num)
    arr[ys, xs] = 0

    return Image.fromarray(arr, mode="L")

def jpeg_degrade(img):
    bio = io.BytesIO()
    quality = random.randint(15, 60)

    img.save(bio, format="JPEG", quality=quality)
    bio.seek(0)

    return Image.open(bio).convert("L")

def random_low_resolution(img):
    w, h = img.size

    if w < 10 or h < 10:
        return img

    scale = random.uniform(0.4, 0.8)

    small_w = max(4, int(w * scale))
    small_h = max(4, int(h * scale))

    small = img.resize(
        (small_w, small_h),
        Image.BILINEAR
    )

    img = small.resize(
        (w, h),
        Image.BICUBIC
    )

    return img

def random_occlusion(img):
    draw = ImageDraw.Draw(img)
    w, h = img.size

    for _ in range(random.randint(0, 2)):
        if random.random() < 0.35:
            x1 = random.randint(0, w - 1)
            y1 = random.randint(0, h - 1)

            x2 = min(w, x1 + random.randint(5, 18))
            y2 = min(h, y1 + random.randint(2, 8))

            color = random.randint(120, 255)

            draw.rectangle([x1, y1, x2, y2], fill=color)

    return img

def crop_to_text_area(img):
    arr = np.array(img)

    mask = arr < 230

    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return img

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    pad_x = random.randint(2, 10)
    pad_y = random.randint(1, 5)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(arr.shape[1] - 1, x2 + pad_x)
    y2 = min(arr.shape[0] - 1, y2 + pad_y)

    return img.crop((x1, y1, x2 + 1, y2 + 1))

def fit_height_keep_ratio(img, target_h=IMG_H, min_w=MIN_W, max_w=MAX_W):
    w, h = img.size

    new_w = max(1, int(w * target_h / h))
    new_w = max(min_w, min(new_w, max_w))

    return img.resize((new_w, target_h), Image.BICUBIC)

def render_text_image(text, font_paths):
    bg = generate_clean_background(CANVAS_W, CANVAS_H)
    font = get_font(font_paths)

    text_mask = create_text_layer(text, font, (CANVAS_W, CANVAS_H))
    img = blend_text(bg, text_mask)

    if random.random() < 0.6:
        img = random_perspective(img)

    img = random_affine(img)

    if random.random() < 0.55:
        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(0.5, 1.8)
            )
        )

    if random.random() < 0.15:
        img = img.filter(ImageFilter.SHARPEN)

    if random.random() < 0.40:
        img = add_gaussian_noise(
            img,
            sigma=random.uniform(2, 12)
        )

    if random.random() < 0.15:
        img = add_salt_pepper_noise(
            img,
            amount=random.uniform(0.001, 0.004)
        )

    if random.random() < 0.35:
        img = jpeg_degrade(img)

    if random.random() < 0.35:
        img = ImageEnhance.Contrast(img).enhance(
            random.uniform(0.65, 1.45)
        )

    if random.random() < 0.30:
        img = ImageEnhance.Brightness(img).enhance(
            random.uniform(0.70, 1.25)
        )

    img = random_occlusion(img)

    if random.random() < 0.25:
        img = random_low_resolution(img)

    img = crop_to_text_area(img)
    img = fit_height_keep_ratio(img)

    return img

def save_split(split_name, count, font_paths, start_idx):
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    img_dir = os.path.join(split_dir, "images")

    ensure_dir(img_dir)

    label_path = os.path.join(split_dir, "labels.txt")
    lines = []

    for i in range(count):
        text = random_word()
        img = render_text_image(text, font_paths)

        filename = f"image_{start_idx + i:06d}.png"

        img.save(os.path.join(img_dir, filename))

        lines.append(f"{filename} {text}")

        if (i + 1) % 1000 == 0:
            print(f"{split_name}: {i + 1}/{count}")

    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"{split_name} saved: {count}")

def main():
    reset_dir(OUTPUT_DIR)

    font_paths = load_fonts(FONTS_DIR)
    print(f"fonts loaded: {len(font_paths)}")

    save_split("train", TRAIN_COUNT, font_paths, 0)
    save_split("val", VAL_COUNT, font_paths, TRAIN_COUNT)
    save_split("test", TEST_COUNT, font_paths, TRAIN_COUNT + VAL_COUNT)

    print("realistic synthetic dataset generation finished.")
    print("output:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
