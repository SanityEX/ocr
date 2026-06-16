import os
import random

SYNTH_TXT = r"D:\mnist_project\ocr1\data_realprint_realistic_100k\train\labels.txt"
SYNTH_IMG_DIR = r"D:\mnist_project\ocr1\data_realprint_realistic_100k\train\images"

IIIT_TRAIN_TXT = r"D:\mnist_project\ocr1\iiit5k_alnum\train\labels.txt"
IIIT_TRAIN_DIR = r"D:\mnist_project\ocr1\iiit5k_alnum\train\images"

IIIT_VAL_TXT = r"D:\mnist_project\ocr1\iiit5k_alnum\val\labels.txt"
IIIT_VAL_DIR = r"D:\mnist_project\ocr1\iiit5k_alnum\val\images"

OUTPUT_TXT = r"D:\mnist_project\ocr1\merge_train_v2.txt"

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

            label = label.strip().upper()
            img_path = os.path.join(img_dir, os.path.basename(path))
            img_path = img_path.replace("\\", "/")

            if os.path.exists(img_path):
                samples.append((img_path, label))

    return samples

def main():
    synth = read_labels(SYNTH_TXT, SYNTH_IMG_DIR)
    iiit_train = read_labels(IIIT_TRAIN_TXT, IIIT_TRAIN_DIR)
    iiit_val = read_labels(IIIT_VAL_TXT, IIIT_VAL_DIR)

    all_samples = synth + iiit_train + iiit_val
    random.shuffle(all_samples)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        for path, label in all_samples:
            f.write(f"{path}\t{label}\n")

    print("=" * 50)
    print("Synthetic:", len(synth))
    print("IIIT train:", len(iiit_train))
    print("IIIT val:", len(iiit_val))
    print("Total:", len(all_samples))
    print("Saved:", OUTPUT_TXT)
    print("=" * 50)

if __name__ == "__main__":
    main()
