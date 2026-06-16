import os
import random
import shutil

from utils import CHARS

RAW_DIR = "IIIT5K-Word_V3.0/IIIT5K"
OUT_DIR = "iiit5k_alnum"

TRAIN_RATIO = 0.9  # train -> 90%, val -> 10%


def reset_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def clean_label(text: str) -> str:
    allowed = set(CHARS)
    return "".join([c for c in text.strip() if c in allowed])


def load_labels(label_file):
    samples = []
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            filename, text = line.split(maxsplit=1)
            text = clean_label(text)

            if len(text) == 0:
                continue

            samples.append((filename, text))

    return samples


def save_split(samples, src_image_dir, dst_root):
    dst_image_dir = os.path.join(dst_root, "images")
    ensure_dir(dst_image_dir)

    label_file = os.path.join(dst_root, "labels.txt")
    lines = []

    for filename, text in samples:
        src = os.path.join(src_image_dir, filename)
        dst = os.path.join(dst_image_dir, filename)

        if os.path.exists(src):
            shutil.copy(src, dst)
            lines.append(f"{filename} {text}")
        else:
            print(f"[WARN] image not found: {src}")

    with open(label_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    reset_dir(OUT_DIR)

    raw_train_img = os.path.join(RAW_DIR, "train")
    raw_train_label = os.path.join(RAW_DIR, "train", "labels.txt")

    raw_test_img = os.path.join(RAW_DIR, "test")
    raw_test_label = os.path.join(RAW_DIR, "test", "labels.txt")

    train_samples = load_labels(raw_train_label)
    test_samples = load_labels(raw_test_label)

    random.shuffle(train_samples)

    split_idx = int(len(train_samples) * TRAIN_RATIO)
    train_final = train_samples[:split_idx]
    val_final = train_samples[split_idx:]

    save_split(train_final, raw_train_img, os.path.join(OUT_DIR, "train"))
    save_split(val_final, raw_train_img, os.path.join(OUT_DIR, "val"))
    save_split(test_samples, raw_test_img, os.path.join(OUT_DIR, "test"))

    print("prepare finished.")
    print(f"train: {len(train_final)}")
    print(f"val:   {len(val_final)}")
    print(f"test:  {len(test_samples)}")


if __name__ == "__main__":
    main()