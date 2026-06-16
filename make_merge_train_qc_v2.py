import os
import random

QC_TXT = r"D:\mnist_project\ocr1\char_occlusion_dataset_qc\labels.txt"
QC_DIR = r"D:\mnist_project\ocr1\char_occlusion_dataset_qc\images"

ISTD_TXT = r"D:\mnist_project\ocr1\istd_train_10_20_30.txt"

NORMAL_TXT = r"D:\mnist_project\ocr1\data_realprint_realistic_100k\train\labels.txt"
NORMAL_DIR = r"D:\mnist_project\ocr1\data_realprint_realistic_100k\train\images"

OUTPUT_TXT = r"D:\mnist_project\ocr1\merge_train_qc_v2.txt"

QC_LIMIT = 30000
NORMAL_LIMIT = 10000
ISTD_REPEAT = 3

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
                path = os.path.join(
                    img_dir,
                    os.path.basename(path)
                )

            path = path.replace("\\", "/")

            if os.path.exists(path):
                samples.append((path, label))

    return samples

def main():

    qc = read_labels(QC_TXT, QC_DIR)

    random.shuffle(qc)

    qc = qc[:QC_LIMIT]

    normal = read_labels(
        NORMAL_TXT,
        NORMAL_DIR
    )

    random.shuffle(normal)

    normal = normal[:NORMAL_LIMIT]

    istd = read_labels(ISTD_TXT, None)

    istd = istd * ISTD_REPEAT

    all_samples = (
        qc +
        normal +
        istd
    )

    random.shuffle(all_samples)

    with open(
        OUTPUT_TXT,
        "w",
        encoding="utf-8"
    ) as f:

        for path, label in all_samples:

            f.write(
                f"{path}\t{label}\n"
            )

    print("=" * 50)
    print("qc:", len(qc))
    print("normal:", len(normal))
    print("istd:", len(istd))
    print("total:", len(all_samples))
    print("saved:", OUTPUT_TXT)
    print("=" * 50)

if __name__ == "__main__":
    main()
