import os
import random
import shutil

TRAIN_DIR = os.path.join("data", "train")
IMAGE_DIR = os.path.join(TRAIN_DIR, "images")
LABEL_FILE = os.path.join(TRAIN_DIR, "labels.txt")

VAL_DIR = os.path.join("data", "val")
TEST_DIR = os.path.join("data", "test")

val_ratio = 0.1
test_ratio = 0.1

with open(LABEL_FILE, "r", encoding="utf-8") as f:
    lines = [line for line in f if line.strip()]

random.shuffle(lines)

total = len(lines)
val_count = int(total * val_ratio)
test_count = int(total * test_ratio)
train_count = total - val_count - test_count

new_train_lines = lines[:train_count]
val_lines = lines[train_count:train_count + val_count]
test_lines = lines[train_count + val_count:]

os.makedirs(os.path.join(VAL_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(TEST_DIR, "images"), exist_ok=True)

with open(LABEL_FILE, "w", encoding="utf-8") as f:
    f.writelines(new_train_lines)

with open(os.path.join(VAL_DIR, "labels.txt"), "w", encoding="utf-8") as f:
    for line in val_lines:
        f.write(line)
        img_name = line.strip().split()[0]
        src = os.path.join(IMAGE_DIR, img_name)
        dst = os.path.join(VAL_DIR, "images", img_name)
        shutil.copy(src, dst)

with open(os.path.join(TEST_DIR, "labels.txt"), "w", encoding="utf-8") as f:
    for line in test_lines:
        f.write(line)
        img_name = line.strip().split()[0]
        src = os.path.join(IMAGE_DIR, img_name)
        dst = os.path.join(TEST_DIR, "images", img_name)
        shutil.copy(src, dst)

print("拆分完成")
print("train:", len(new_train_lines))
print("val:", len(val_lines))
print("test:", len(test_lines))
