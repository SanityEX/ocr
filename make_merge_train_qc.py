import os
import random


QC_TXT = r"D:\mnist_project\ocr1\char_occlusion_dataset_qc\labels.txt"
QC_DIR = r"D:\mnist_project\ocr1\char_occlusion_dataset_qc\images"

ISTD_TXT = r"D:\mnist_project\ocr1\istd_train_10_20_30.txt"

OUTPUT_TXT = r"D:\mnist_project\ocr1\merge_train_qc.txt"

QC_LIMIT = 30000
ISTD_REPEAT = 2


def read_labels(txt_path, img_dir=None):
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

            if img_dir is not None:
                path = os.path.join(img_dir, os.path.basename(path))

            path = path.replace("\\", "/")

            if os.path.exists(path):
                samples.append((path, label))

    return samples


def main():
    qc = read_labels(QC_TXT, QC_DIR)
    random.shuffle(qc)
    qc = qc[:QC_LIMIT]

    istd = read_labels(ISTD_TXT, None)
    istd = istd * ISTD_REPEAT

    all_samples = qc + istd
    random.shuffle(all_samples)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        for path, label in all_samples:
            f.write(f"{path}\t{label}\n")

    print("=" * 50)
    print("qc:", len(qc))
    print("istd:", len(istd))
    print("total:", len(all_samples))
    print("saved:", OUTPUT_TXT)
    print("=" * 50)


if __name__ == "__main__":
    main()