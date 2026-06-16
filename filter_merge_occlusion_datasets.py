import os
import random
import shutil
import string
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance


SAVE_DIR = r"D:\mnist_project\ocr1\char_occlusion_dataset_qc"
IMG_DIR = os.path.join(SAVE_DIR, "images")
LABEL_PATH = os.path.join(SAVE_DIR, "labels.txt")
META_PATH = os.path.join(SAVE_DIR, "metadata.txt")

FONT_DIR = r"D:\mnist_project\ocr1\fonts"

TARGET_COUNT = 30000
MAX_TRIALS = 150000

IMG_W = 192
IMG_H = 48
FONT_SIZE_MIN = 24
FONT_SIZE_MAX = 36

MIN_ERASE_RATIO = 0.15
MAX_ERASE_RATIO = 0.55
MIN_REMAIN_RATIO = 0.35

WORDS = [
    "AIRPORT", "AVENUE", "BANK", "BEACH", "BLACK", "BLUE", "BOOK", "BRIDGE",
    "BUILDING", "CAFE", "CENTER", "CENTRAL", "CHOCOLATE", "CITY", "CLASSIC",
    "COFFEE", "COMPANY", "DIGITAL", "DISPLAY", "DRIVE", "ENTRANCE", "EXIT",
    "FAIRPRICE", "FAMILY", "FINEST", "FOOD", "FOUNTAIN", "GARDEN", "GIANT",
    "GREEN", "HALL", "HOTEL", "HYPERMARKET", "IMAGE", "INFORMATION",
    "INTERNATIONAL", "LIBRARY", "MARKET", "MEETING", "MITSUBISHI", "MODEL",
    "MONITOR", "MUSIC", "OFFICE", "ORANGE", "ORIENTAL", "PARKING", "PASSAGES",
    "PHONE", "POINT", "PRICE", "PRIVATE", "PROJECT", "QUALITY", "READER",
    "REPUBLIC", "RESTAURANT", "RESTROOM", "REVIEW", "SAVINGS", "SCHOOL",
    "SCREEN", "SERVICE", "SHOPPING", "STATION", "STREET", "SYSTEM", "THEATRE",
    "TOWN", "TRAIN", "TRAVEL", "UNIVERSITY", "VALUE", "WEALTH", "WINDOW",
    "WORLD", "YELLOW", "BOOKING", "BUTTON", "CITIBANK", "COLOUR", "CUSTOMER",
    "ESPLANADE", "HOSPITAL", "LANGUAGE", "MANDARIN", "NETWORK", "PLATFORM",
    "RESEARCH", "SECURITY", "STANDARD", "TRANSFER", "TRANSPORT", "WEBSITE"
]

MASK_MODES = [
    "top",
    "middle",
    "bottom",
    "left",
    "right",
    "thin_line",
    "vertical_cut",
    "small_patch"
]


def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def load_fonts(font_dir):
    fonts = []
    if os.path.exists(font_dir):
        for f in os.listdir(font_dir):
            if f.lower().endswith((".ttf", ".otf")):
                fonts.append(os.path.join(font_dir, f))
    return fonts


def get_font(fonts):
    size = random.randint(FONT_SIZE_MIN, FONT_SIZE_MAX)
    if fonts:
        return ImageFont.truetype(random.choice(fonts), size)
    return ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", size)


def random_word():
    if random.random() < 0.9:
        return random.choice(WORDS)
    return "".join(random.choice(string.ascii_uppercase) for _ in range(random.randint(4, 9)))


def draw_word(text, font):
    img = Image.new("L", (IMG_W, IMG_H), 255)
    draw = ImageDraw.Draw(img)

    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    x = max(2, (IMG_W - tw) // 2)
    y = max(2, (IMG_H - th) // 2)

    draw.text((x, y), text, fill=random.randint(0, 35), font=font)
    return img


def make_mask_box():
    mode = random.choice(MASK_MODES)

    if mode == "top":
        x = random.randint(0, IMG_W - 30)
        y = random.randint(0, 14)
        w = random.randint(16, 55)
        h = random.randint(4, 9)

    elif mode == "middle":
        x = random.randint(0, IMG_W - 30)
        y = random.randint(IMG_H // 3, IMG_H // 2)
        w = random.randint(16, 55)
        h = random.randint(4, 11)

    elif mode == "bottom":
        x = random.randint(0, IMG_W - 30)
        y = random.randint(IMG_H - 18, IMG_H - 7)
        w = random.randint(16, 55)
        h = random.randint(4, 9)

    elif mode == "left":
        x = random.randint(0, 50)
        y = random.randint(6, IMG_H - 20)
        w = random.randint(6, 20)
        h = random.randint(12, 32)

    elif mode == "right":
        x = random.randint(IMG_W - 55, IMG_W - 15)
        y = random.randint(6, IMG_H - 20)
        w = random.randint(6, 20)
        h = random.randint(12, 32)

    elif mode == "thin_line":
        x = random.randint(0, IMG_W - 50)
        y = random.randint(8, IMG_H - 10)
        w = random.randint(40, 100)
        h = random.randint(2, 5)

    elif mode == "vertical_cut":
        x = random.randint(10, IMG_W - 20)
        y = random.randint(4, IMG_H - 25)
        w = random.randint(4, 9)
        h = random.randint(18, 38)

    else:
        x = random.randint(0, IMG_W - 25)
        y = random.randint(5, IMG_H - 20)
        w = random.randint(10, 30)
        h = random.randint(7, 18)

    return mode, (x, y, x + w, y + h)


def apply_occlusion(original):
    occ = original.copy()
    draw = ImageDraw.Draw(occ)

    mask_count = random.randint(1, 2)
    boxes = []

    for _ in range(mask_count):
        mode, box = make_mask_box()
        draw.rectangle(box, fill=255)
        boxes.append((mode, box))

    return occ, boxes


def calc_quality(original, occluded):
    orig_arr = np.array(original)
    occ_arr = np.array(occluded)

    orig_ink = orig_arr < 180
    occ_ink = occ_arr < 180

    orig_count = np.sum(orig_ink)
    remain_count = np.sum(occ_ink)

    if orig_count == 0:
        return 0, 0

    erased = np.logical_and(orig_ink, np.logical_not(occ_ink))
    erased_count = np.sum(erased)

    erase_ratio = erased_count / orig_count
    remain_ratio = remain_count / orig_count

    return erase_ratio, remain_ratio


def degrade(img):
    if random.random() < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.7)))

    if random.random() < 0.25:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.20))

    if random.random() < 0.15:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.90, 1.10))

    return img


def main():
    reset_dir(SAVE_DIR)
    os.makedirs(IMG_DIR, exist_ok=True)

    fonts = load_fonts(FONT_DIR)

    labels = []
    meta = []

    kept = 0
    trials = 0

    while kept < TARGET_COUNT and trials < MAX_TRIALS:
        trials += 1

        text = random_word()
        font = get_font(fonts)

        original = draw_word(text, font)
        occluded, boxes = apply_occlusion(original)

        erase_ratio, remain_ratio = calc_quality(original, occluded)

        if erase_ratio < MIN_ERASE_RATIO:
            continue

        if erase_ratio > MAX_ERASE_RATIO:
            continue

        if remain_ratio < MIN_REMAIN_RATIO:
            continue

        occluded = degrade(occluded)

        filename = f"qc_occ_{kept:06d}.png"
        occluded.save(os.path.join(IMG_DIR, filename))

        labels.append(f"{filename}\t{text}")
        meta.append(f"{filename}\t{text}\terase={erase_ratio:.4f}\tremain={remain_ratio:.4f}\tboxes={boxes}")

        kept += 1

        if kept % 500 == 0:
            print(f"kept: {kept}/{TARGET_COUNT} | trials: {trials}")

    with open(LABEL_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(labels))

    with open(META_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(meta))

    print("=" * 50)
    print("DONE")
    print("kept:", kept)
    print("trials:", trials)
    print("save:", SAVE_DIR)
    print("=" * 50)


if __name__ == "__main__":
    main()