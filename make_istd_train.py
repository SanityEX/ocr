import os
import random

ROOT = r"D:\mnist_project\ocr1\recognition\recognition"
GT_TXT = r"D:\mnist_project\ocr1\recognition\recognition\gt_recognition.txt"

LEVELS = ["10", "20", "30"]

OUT_TXT = r"D:\mnist_project\ocr1\istd_train_10_20_30.txt"

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
            label = label.replace('"', "").upper()

            data[name] = label

    return data

def main():
    gt = load_gt(GT_TXT)
    samples = []

    for level in LEVELS:
        img_dir = os.path.join(ROOT, level)

        for filename in os.listdir(img_dir):
            if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            gt_name = filename.replace("img_", "word_")
            gt_name = os.path.splitext(gt_name)[0] + ".png"

            if gt_name not in gt:
                continue

            img_path = os.path.join(img_dir, filename).replace("\\", "/")
            label = gt[gt_name]

            samples.append((img_path, label))

    random.shuffle(samples)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        for path, label in samples:
            f.write(f"{path}\t{label}\n")

    print("=" * 50)
    print("levels:", LEVELS)
    print("total:", len(samples))
    print("saved:", OUT_TXT)
    print("=" * 50)

if __name__ == "__main__":
    main()
