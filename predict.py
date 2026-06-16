import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import CRNN
from utils import decode_ctc, CHARS


# ========= 配置 =========
WEIGHTS = "best_crnn_iiit5k_mix2_acc.pth"   # 改成你的模型
IMG_PATH = "data/test/images/point.png"                       # 改成你的图片


IMG_H = 48
MIN_W = 48
MAX_W = 256


# ========= 预处理 =========
def resize_keep_ratio(image):
    w, h = image.size
    new_w = max(1, int(w * IMG_H / h))
    new_w = max(MIN_W, min(new_w, MAX_W))
    image = image.resize((new_w, IMG_H), Image.BICUBIC)
    return image


def preprocess(image_path):
    image = Image.open(image_path).convert("L")
    image = resize_keep_ratio(image)

    tensor = transforms.ToTensor()(image)   # [1, H, W]
    tensor = tensor.unsqueeze(0)            # [1, 1, H, W]

    return tensor


# ========= decode =========
def greedy_decode(logits):
    preds = logits.argmax(dim=2)   # [T, B]
    preds = preds.permute(1, 0)    # [B, T]

    results = []
    for seq in preds:
        indices = []
        prev = None
        for idx in seq.cpu().tolist():
            if idx != 0 and idx != prev:
                indices.append(idx)
            prev = idx
        results.append(decode_ctc(indices))

    return results


# ========= 主函数 =========
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = CRNN(num_classes=len(CHARS) + 1).to(device)

    print("loading:", WEIGHTS)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.eval()

    image_tensor = preprocess(IMG_PATH).to(device)

    with torch.no_grad():
        logits = model(image_tensor)

    preds = greedy_decode(logits)

    print("\n===== RESULT =====")
    print("image:", IMG_PATH)
    print("pred :", preds[0])


if __name__ == "__main__":
    main()