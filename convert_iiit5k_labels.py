import os
from scipy.io import loadmat

ROOT = "IIIT5K-Word_V3.0/IIIT5K"

def unwrap(x):
    while hasattr(x, "__len__") and not isinstance(x, str):
        if len(x) == 1:
            x = x[0]
        else:
            break
    return str(x)

def convert(mat_name, key_name, split_name):
    mat_path = os.path.join(ROOT, mat_name)
    split_root = os.path.join(ROOT, split_name)

    data = loadmat(mat_path)
    arr = data[key_name]

    lines = []

    for i in range(arr.shape[1]):
        item = arr[0, i]

        try:
            img_name = unwrap(item[0]).strip().replace("\\", "/")
            label = unwrap(item[1]).strip()

            if img_name.startswith("train/") or img_name.startswith("test/"):
                full_path = os.path.join(ROOT, img_name)
                save_name = os.path.basename(img_name)
            else:
                full_path = os.path.join(split_root, img_name)
                save_name = img_name

            if not os.path.exists(full_path):
                print(f"[WARN] not found: {full_path}")
                continue

            lines.append(f"{save_name} {label}")

        except Exception as e:
            print(f"[ERROR] index {i}: {e}")

    save_path = os.path.join(split_root, "labels.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n✔ saved: {save_path}")
    print(f"total: {len(lines)}")

def main():
    convert("traindata.mat", "traindata", "train")
    convert("testdata.mat", "testdata", "test")

if __name__ == "__main__":
    main()
