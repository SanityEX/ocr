import os
import random


NORMAL_TXT = r"D:\mnist_project\ocr1\data_realprint_realistic_100k\train\labels.txt"
NORMAL_DIR = r"D:\mnist_project\ocr1\data_realprint_realistic_100k\train\images"

OCC_TXT = r"D:\mnist_project\ocr1\word_occluded_10k\labels.txt"
OCC_DIR = r"D:\mnist_project\ocr1\word_occluded_10k\images"

IIIT_TRAIN_TXT = r"D:\mnist_project\ocr1\iiit5k_alnum\train\labels.txt"
IIIT_TRAIN_DIR = r"D:\mnist_project\ocr1\iiit5k_alnum\train\images"

OUTPUT_TXT = r"D:\mnist_project\ocr1\merge_train_v3.txt"


def read_labels(txt_path, img_dir):
    samples = []

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if "\t" in line:
                path, label = line.split("\t", 1)
            else:
                parts = line.split(maxsplit=1)

                if len(parts) != 2:
                    continue

                path, label = parts

            img_path = os.path.join(img_dir, os.path.basename(path))
            img_path = img_path.replace("\\", "/")
            label = label.strip().upper()

            if os.path.exists(img_path):
                samples.append((img_path, label))

    return samples


def main():
    normal = read_labels(NORMAL_TXT, NORMAL_DIR)
    occ = read_labels(OCC_TXT, OCC_DIR)
    iiit = read_labels(IIIT_TRAIN_TXT, IIIT_TRAIN_DIR)

    all_samples = normal + occ + iiit
    random.shuffle(all_samples)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        for path, label in all_samples:
            f.write(f"{path}\t{label}\n")

    print("=" * 50)
    print("normal:", len(normal))
    print("occluded:", len(occ))
    print("iiit:", len(iiit))
    print("total:", len(all_samples))
    print("saved:", OUTPUT_TXT)
    print("=" * 50)


if __name__ == "__main__":
    main()