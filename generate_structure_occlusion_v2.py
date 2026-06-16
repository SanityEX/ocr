import os
import random
import string
import shutil
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance


SAVE_DIR = r"D:\mnist_project\ocr1\char_occlusion_dataset_v2"
IMG_DIR = os.path.join(SAVE_DIR, "images")
LABEL_PATH = os.path.join(SAVE_DIR, "labels.txt")

FONT_DIR = r"D:\mnist_project\ocr1\fonts"

NUM_SAMPLES = 30000

IMG_W = 192
IMG_H = 48
FONT_SIZE_MIN = 24
FONT_SIZE_MAX = 36

WORDS = [
    "ABOUT", "ABOVE", "ACADEMY", "ACCESS", "ACCOUNT", "ACTION", "ADDRESS", "ADVANCE",
    "AIRPORT", "ALERT", "ALPHA", "AMERICAN", "APPLE", "ARCADE", "AREA", "AROUND",
    "ARRIVAL", "ASIA", "AUTO", "AVENUE", "BANK", "BEACH", "BEAUTY", "BLACK",
    "BLUE", "BOOK", "BORDER", "BRANCH", "BRIDGE", "BUILDING", "BUSINESS", "BUTTON",
    "CAFE", "CAMERA", "CAPITAL", "CAR", "CARD", "CARE", "CENTER", "CENTRAL",
    "CHANNEL", "CHECK", "CHOCOLATE", "CINEMA", "CIRCLE", "CITY", "CLASSIC", "CLINIC",
    "CLOSE", "COFFEE", "COLLEGE", "COLOR", "COMPANY", "COMPUTER", "CONTACT", "CONTROL",
    "CORNER", "COUNTER", "COURSE", "CREDIT", "DAILY", "DATA", "DELIVERY", "DESIGN",
    "DIGITAL", "DINNER", "DIRECT", "DISPLAY", "DOCTOR", "DOOR", "DRIVE", "EAST",
    "EDITOR", "EDUCATION", "ELECTRIC", "ENERGY", "ENTER", "ENTRANCE", "ERROR", "EVENT",
    "EXCHANGE", "EXIT", "EXPRESS", "FACTORY", "FAIRPRICE", "FAMILY", "FASHION", "FAST",
    "FEATURE", "FILTER", "FINEST", "FLOOR", "FOOD", "FOUNTAIN", "FRONT", "FUTURE",
    "GALLERY", "GARDEN", "GENERAL", "GIANT", "GLOBAL", "GREEN", "GROUP", "GUIDE",
    "HALL", "HEALTH", "HELLO", "HELP", "HIGH", "HOME", "HOTEL", "HOUSE",
    "HYPER", "HYPERMARKET", "IMAGE", "INFORMATION", "INPUT", "INSIDE", "INTERNATIONAL", "INTERNET",
    "ISLAND", "JAPAN", "JOURNEY", "JUSTICE", "KENKO", "KITCHEN", "LABEL", "LABORATORY",
    "LARGE", "LEADER", "LEAVES", "LEVEL", "LIBRARY", "LIGHT", "LIMITED", "LITTLE",
    "LOCAL", "LONDON", "LUNCH", "MARKET", "MASTER", "MEETING", "MESSAGE", "MITSUBISHI",
    "MODEL", "MOLLY", "MONITOR", "MORNING", "MUSIC", "NATIONAL", "NETWORK", "NIGHT",
    "NORTH", "OFFICE", "OPEN", "ORANGE", "ORIENTAL", "OUTPUT", "PACKAGE", "PARKING",
    "PASSAGE", "PASSAGES", "PAYMENT", "PHONE", "PHOTO", "PICTURE", "POINT", "POWER",
    "PRICE", "PRIVATE", "PROJECT", "PUBLIC", "QUALITY", "QUEEN", "READER", "REPUBLIC",
    "RESTAURANT", "RESTAURANTS", "RESULT", "REVIEW", "RIVER", "ROAD", "ROOM", "SAVINGS",
    "SCHOOL", "SCREEN", "SERVICE", "SHOP", "SIGNAL", "SILVER", "SOUTH", "SPACE",
    "SPECIAL", "SPORT", "STATION", "STORE", "STREET", "STUDIO", "SYSTEM", "TABLE",
    "TECHNOLOGY", "TELEPHONE", "THEATRE", "TICKET", "TOKYO", "TOWN", "TRAIN", "TRAVEL",
    "UNION", "UNIVERSAL", "UNIVERSITY", "VALUE", "VIDEO", "VIEW", "VISION", "VISITOR",
    "WATER", "WEALTH", "WELCOME", "WEST", "WINDOW", "WINTER", "WORLD", "YELLOW",
    "BOOKING", "BOTTLE", "BUDGET", "BUSSTOP", "CAFETERIA", "CALENDAR", "CARPARK",
    "CHARCOAL", "CITIBANK", "COLOUR", "COMMUNITY", "CUSTOMER", "DEPARTMENT", "DIRECTION",
    "DISCOUNT", "DOWNLOAD", "ELEVATOR", "EMERGENCY", "EMPLOYEE", "ENGINEER", "ESPLANADE",
    "EXPERIENCE", "GOVERNMENT", "GRADUATE", "HOSPITAL", "INSURANCE", "LANGUAGE",
    "LOCATION", "MAGAZINE", "MANDARIN", "MANAGER", "MEDICAL", "MEMBER", "OFFICIAL",
    "OPERATION", "PLATFORM", "PRODUCT", "PROGRAM", "REGISTER", "RESEARCH", "RESERVED",
    "RESTROOM", "SECURITY", "SHOPPING", "STANDARD", "STUDENT", "TERMINAL", "TRANSFER",
    "TRANSPORT", "VEHICLE", "WEBSITE"
]

CONFUSION_WORDS = [
    "OFFICE", "BOOK", "FOOD", "SCHOOL", "LOOK", "GOOD", "POOL", "ROOM",
    "BORDER", "BROOK", "BOTTLE", "BUTTON", "DIGITAL", "DISPLAY", "LIMITED",
    "LITTLE", "LIFT", "LIGHT", "EXIT", "FINEST", "FILTER", "VALUE", "AVENUE",
    "BAY", "BANK", "BAR", "BUD", "BLUE", "BLACK", "BRIDGE", "BUILDING"
]

MASK_MODES = [
    "top",
    "middle",
    "bottom",
    "left",
    "right",
    "thin_line",
    "vertical_cut",
    "diagonal",
    "small_patch"
]


def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def load_fonts(font_dir):
    fonts = []

    if not os.path.exists(font_dir):
        return fonts

    for file in os.listdir(font_dir):
        low = file.lower()

        if low.endswith(".ttf") or low.endswith(".otf"):
            fonts.append(os.path.join(font_dir, file))

    return fonts


def get_font(fonts):
    size = random.randint(FONT_SIZE_MIN, FONT_SIZE_MAX)

    if fonts:
        return ImageFont.truetype(random.choice(fonts), size)

    return ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", size)


def random_word():
    p = random.random()

    if p < 0.75:
        return random.choice(WORDS)

    if p < 0.90:
        return random.choice(CONFUSION_WORDS)

    length = random.randint(4, 10)

    return "".join(random.choice(string.ascii_uppercase) for _ in range(length))


def draw_word(text, font):
    img = Image.new("L", (IMG_W, IMG_H), color=255)
    draw = ImageDraw.Draw(img)

    bbox = draw.textbbox((0, 0), text, font=font)

    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = max(2, (IMG_W - text_w) // 2)
    y = max(2, (IMG_H - text_h) // 2 - random.randint(0, 2))

    draw.text((x, y), text, fill=random.randint(0, 40), font=font)

    return img


def erase_structure(img):
    draw = ImageDraw.Draw(img)

    mode = random.choice(MASK_MODES)

    mask_count = random.randint(1, 3)

    for _ in range(mask_count):
        if mode == "top":
            x = random.randint(0, IMG_W - 20)
            y = random.randint(0, 12)
            w = random.randint(12, 45)
            h = random.randint(3, 8)

        elif mode == "middle":
            x = random.randint(0, IMG_W - 20)
            y = random.randint(IMG_H // 3, IMG_H // 2)
            w = random.randint(12, 45)
            h = random.randint(3, 10)

        elif mode == "bottom":
            x = random.randint(0, IMG_W - 20)
            y = random.randint(IMG_H - 16, IMG_H - 6)
            w = random.randint(12, 45)
            h = random.randint(3, 8)

        elif mode == "left":
            x = random.randint(0, 40)
            y = random.randint(5, IMG_H - 15)
            w = random.randint(5, 18)
            h = random.randint(10, 30)

        elif mode == "right":
            x = random.randint(IMG_W - 45, IMG_W - 12)
            y = random.randint(5, IMG_H - 15)
            w = random.randint(5, 18)
            h = random.randint(10, 30)

        elif mode == "thin_line":
            x = random.randint(0, IMG_W - 30)
            y = random.randint(8, IMG_H - 10)
            w = random.randint(30, 90)
            h = random.randint(2, 4)

        elif mode == "vertical_cut":
            x = random.randint(10, IMG_W - 20)
            y = random.randint(4, IMG_H - 20)
            w = random.randint(3, 8)
            h = random.randint(15, 36)

        elif mode == "diagonal":
            x = random.randint(0, IMG_W - 40)
            y = random.randint(5, IMG_H - 20)
            w = random.randint(25, 55)
            h = random.randint(4, 9)

            patch = Image.new("L", (w, h), 255)
            patch = patch.rotate(
                random.choice([-25, -15, 15, 25]),
                expand=True,
                fillcolor=255
            )

            img.paste(patch, (x, y))
            continue

        else:
            x = random.randint(0, IMG_W - 18)
            y = random.randint(4, IMG_H - 14)
            w = random.randint(8, 25)
            h = random.randint(5, 15)

        draw.rectangle([x, y, x + w, y + h], fill=255)

    return img


def degrade(img):
    if random.random() < 0.35:
        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(0.2, 0.8)
            )
        )

    if random.random() < 0.30:
        img = ImageEnhance.Contrast(img).enhance(
            random.uniform(0.75, 1.25)
        )

    if random.random() < 0.20:
        img = ImageEnhance.Brightness(img).enhance(
            random.uniform(0.85, 1.15)
        )

    return img


def main():
    reset_dir(SAVE_DIR)
    os.makedirs(IMG_DIR, exist_ok=True)

    fonts = load_fonts(FONT_DIR)

    labels = []

    for i in range(NUM_SAMPLES):
        text = random_word()
        font = get_font(fonts)

        img = draw_word(text, font)
        img = erase_structure(img)
        img = degrade(img)

        filename = f"struct_occ_{i:06d}.png"

        img.save(os.path.join(IMG_DIR, filename))

        labels.append(f"{filename}\t{text}")

        if (i + 1) % 500 == 0:
            print(f"generated: {i + 1}/{NUM_SAMPLES}")

    with open(LABEL_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(labels))

    print("=" * 50)
    print("DONE")
    print("images:", IMG_DIR)
    print("labels:", LABEL_PATH)
    print("samples:", NUM_SAMPLES)
    print("words:", len(WORDS))
    print("=" * 50)


if __name__ == "__main__":
    main()