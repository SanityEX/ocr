import os
import random
import shutil
import string
import math
import numpy as np

from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUTPUT_DIR = "data_realprint_alnum"

TRAIN_COUNT = 30000
VAL_COUNT = 3000
TEST_COUNT = 3000

IMG_H = 48
MIN_W = 48
MAX_W = 256
CANVAS_W = 320

FONTS_DIR = "fonts"
FONT_SIZES = [22, 24, 26, 28, 30, 32, 34, 36]

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
    "hangover", "divorce", "blush", "ultimate", "november"
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
        lower = file.lower()
        if lower.endswith(".ttf") or lower.endswith(".otf"):
            fonts.append(os.path.join(fonts_dir, file))
    return fonts

def random_case(word: str) -> str:
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

def random_alnum_string(min_len=3, max_len=14) -> str:
    length = random.randint(min_len, max_len)
    return "".join(random.choices(CHARS, k=length))

def random_word():
    """
    混合生成：
    1. 常用英文词（各种大小写）
    2. 词 + 数字
    3. 纯随机 alnum
    4. 品牌/缩写风格
    """
    p = random.random()

    if p < 0.45:

        word = random.choice(COMMON_WORDS)
        return random_case(word)

    elif p < 0.65:

        word = random.choice(COMMON_WORDS)
        word = random_case(word)
        if random.random() < 0.5:
            return word + str(random.randint(0, 9999))
        else:
            return str(random.randint(0, 9999)) + word

    elif p < 0.85:

        return random_alnum_string(3, 14)

    else:

        length = random.randint(2, 8)
        s = "".join(random.choices(string.ascii_uppercase + string.digits, k=length))
        return s

def add_gaussian_noise(img: Image.Image, sigma=8):
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def add_salt_pepper_noise(img: Image.Image, amount=0.003):
    arr = np.array(img)
    num = int(amount * arr.size)

    ys = np.random.randint(0, arr.shape[0], num)
    xs = np.random.randint(0, arr.shape[1], num)
    arr[ys, xs] = 255

    ys = np.random.randint(0, arr.shape[0], num)
    xs = np.random.randint(0, arr.shape[1], num)
    arr[ys, xs] = 0

    return Image.fromarray(arr)

def random_background():
    """
    做更真实的背景：
    - 纯白/灰白
    - 轻微纹理
    - 渐变感
    """
    bg = random.randint(220, 255)
    canvas = np.ones((IMG_H, CANVAS_W), dtype=np.uint8) * bg

    if random.random() < 0.5:
        grad = np.linspace(0, random.randint(-20, 20), CANVAS_W).astype(np.int16)
        canvas = np.clip(canvas.astype(np.int16) + grad, 0, 255).astype(np.uint8)

    if random.random() < 0.5:
        texture = np.random.normal(0, random.uniform(2, 8), canvas.shape)
        canvas = np.clip(canvas.astype(np.float32) + texture, 0, 255).astype(np.uint8)

    return Image.fromarray(canvas)

def choose_text_color(bg_mean: float):

    if bg_mean > 160:
        return random.randint(0, 60)
    else:
        return random.randint(180, 255)

def get_font(font_paths):
    if font_paths:
        font_path = random.choice(font_paths)
        font_size = random.choice(FONT_SIZES)
        return ImageFont.truetype(font_path, font_size)
    return ImageFont.load_default()

def draw_text_on_canvas(text: str, font_paths):
    img = random_background()
    draw = ImageDraw.Draw(img)

    font = get_font(font_paths)
    bg_mean = np.array(img).mean()
    fg = choose_text_color(bg_mean)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = random.randint(2, max(2, CANVAS_W - text_w - 2))
    y = random.randint(0, max(0, IMG_H - text_h))

    if random.random() < 0.5:
        cur_x = x
        for ch in text:
            ch_bbox = draw.textbbox((0, 0), ch, font=font)
            ch_w = ch_bbox[2] - ch_bbox[0]
            ch_y = y + random.randint(-1, 1)
            draw.text((cur_x, ch_y), ch, font=font, fill=fg)
            cur_x += ch_w + random.randint(0, 2)
    else:
        draw.text((x, y), text, font=font, fill=fg)

    if random.random() < 0.15:
        draw.text((x + 1, y), text, font=font, fill=fg)

    return img

def apply_affine_like_effects(img: Image.Image):

    if random.random() < 0.7:
        angle = random.uniform(-5, 5)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)

    if random.random() < 0.35:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.0)))

    if random.random() < 0.15:
        img = img.filter(ImageFilter.SHARPEN)

    if random.random() < 0.35:
        img = add_gaussian_noise(img, sigma=random.uniform(2, 10))

    if random.random() < 0.15:
        img = add_salt_pepper_noise(img, amount=random.uniform(0.001, 0.005))

    return img

def crop_to_text_area(img: Image.Image):
    arr = np.array(img)

    mask = arr < 200
    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return img

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    pad_x = random.randint(2, 10)
    pad_y = random.randint(1, 4)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(arr.shape[1] - 1, x2 + pad_x)
    y2 = min(arr.shape[0] - 1, y2 + pad_y)

    return img.crop((x1, y1, x2 + 1, y2 + 1))

def fit_height_keep_ratio(img: Image.Image, target_h=IMG_H, min_w=MIN_W, max_w=MAX_W):
    w, h = img.size
    new_w = max(1, int(w * target_h / h))
    new_w = max(min_w, min(new_w, max_w))
    img = img.resize((new_w, target_h), Image.BICUBIC)
    return img

def render_text_image(text: str, font_paths):
    img = draw_text_on_canvas(text, font_paths)
    img = apply_affine_like_effects(img)
    img = crop_to_text_area(img)
    img = fit_height_keep_ratio(img)
    return img

def save_split(split_name: str, count: int, font_paths, start_idx: int):
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

    print("dataset generation finished.")

if __name__ == "__main__":
    main()
