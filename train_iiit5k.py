import os
import copy
import random
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image, ImageFilter

from dataset import OCRDataset
from model import CRNN
from utils import encode_text, decode_ctc, CHARS


SAVE_WEIGHTS_LOSS = "best_crnn_iiit5k_mix2_loss.pth"
SAVE_WEIGHTS_ACC = "best_crnn_iiit5k_mix2_acc.pth"

IMG_H = 48
MIN_W = 48
MAX_W = 256


def resize_keep_ratio(image, img_h=IMG_H, min_w=MIN_W, max_w=MAX_W):
    w, h = image.size
    new_w = max(1, int(w * img_h / h))
    new_w = max(min_w, min(new_w, max_w))
    image = image.resize((new_w, img_h), Image.BICUBIC)
    return image


def augment_image(image):
    # 轻量旋转
    if random.random() < 0.7:
        angle = random.uniform(-4, 4)
        image = image.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)

    # 轻量平移/缩放/剪切
    if random.random() < 0.5:
        dx = random.uniform(-0.05, 0.05) * image.size[0]
        dy = random.uniform(-0.03, 0.03) * image.size[1]
        image = transforms.functional.affine(
            image,
            angle=0,
            translate=(int(dx), int(dy)),
            scale=random.uniform(0.95, 1.05),
            shear=random.uniform(-3, 3),
            fill=255
        )

    # 模糊
    if random.random() < 0.25:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))

    # 亮度/对比度
    if random.random() < 0.4:
        image = transforms.functional.adjust_brightness(image, random.uniform(0.9, 1.1))
    if random.random() < 0.4:
        image = transforms.functional.adjust_contrast(image, random.uniform(0.9, 1.15))

    return image


def make_collate_fn(is_train=True):
    to_tensor = transforms.ToTensor()

    def collate_fn(batch):
        images_pil, texts = zip(*batch)

        processed = []
        widths = []

        for image in images_pil:
            image = resize_keep_ratio(image)

            if is_train:
                image = augment_image(image)

            tensor = to_tensor(image)   # [1, H, W]
            processed.append(tensor)
            widths.append(tensor.shape[-1])

        max_width = max(widths)

        padded_images = []
        for tensor in processed:
            c, h, w = tensor.shape
            pad_w = max_width - w
            padded = F.pad(tensor, (0, pad_w, 0, 0), value=0.0)
            padded_images.append(padded)

        images = torch.stack(padded_images, dim=0)

        targets = []
        target_lengths = []

        for text in texts:
            encoded = encode_text(text)
            targets.extend(encoded)
            target_lengths.append(len(encoded))

        targets = torch.tensor(targets, dtype=torch.long)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)

        return images, texts, targets, target_lengths

    return collate_fn


@torch.no_grad()
def greedy_decode(logits: torch.Tensor) -> list[str]:
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


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, texts, targets, target_lengths) in enumerate(loader):
        if batch_idx % 20 == 0:
            print(f"train batch: {batch_idx}/{len(loader)}")

        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        logits = model(images)
        log_probs = logits.log_softmax(2)

        input_lengths = torch.full(
            size=(images.size(0),),
            fill_value=log_probs.size(0),
            dtype=torch.long,
            device=device
        )

        loss = criterion(log_probs, targets, input_lengths, target_lengths)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    sample_batches = set(random.sample(range(len(loader)), k=min(3, len(loader))))

    for batch_idx, (images, texts, targets, target_lengths) in enumerate(loader):
        if batch_idx % 20 == 0:
            print(f"val batch: {batch_idx}/{len(loader)}")

        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        logits = model(images)
        log_probs = logits.log_softmax(2)

        input_lengths = torch.full(
            size=(images.size(0),),
            fill_value=log_probs.size(0),
            dtype=torch.long,
            device=device
        )

        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        total_loss += loss.item()

        preds = greedy_decode(logits)

        for i, (pred, gt) in enumerate(zip(preds, texts)):
            if batch_idx in sample_batches and i < 2:
                tag = "OK" if pred == gt else "NG"
                print(f"[sample batch {batch_idx}] {tag} | GT: {gt} | Pred: {pred}")

            if pred == gt:
                correct += 1
            total += 1

    avg_loss = total_loss / len(loader)
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


def load_pretrained_partial(model, device):
    candidates = [
        "best_crnn_iiit5k_dyn_acc.pth",
        "best_crnn_iiit5k_dyn_loss.pth",
        "best_crnn_iiit5k_alnum_acc.pth",
        "best_crnn_iiit5k_alnum_loss.pth",
        "best_crnn_iiit5k_mix_acc.pth",
        "best_crnn_iiit5k_mix_loss.pth",
        "best_crnn_realprint.pth",
        "best_crnn_easy.pth",
    ]

    model_dict = model.state_dict()

    for path in candidates:
        if not os.path.exists(path):
            continue

        print(f"trying to load pretrained weights: {path}")
        checkpoint = torch.load(path, map_location=device)

        matched = {}
        skipped = []

        for k, v in checkpoint.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                matched[k] = v
            else:
                skipped.append(k)

        model_dict.update(matched)
        model.load_state_dict(model_dict)

        print(f"loaded matched layers: {len(matched)}")
        print(f"skipped layers: {len(skipped)}")
        if skipped:
            print("example skipped keys:", skipped[:10])

        return

    print("[WARN] no pretrained weights found, training from scratch.")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 主训练集1：IIIT5K alnum
    iiit_train_dataset = OCRDataset("iiit5k_alnum/train", transform=None)

    # 主训练集2：新的更强合成集
    realprint_alnum_train_dataset = OCRDataset("data_realprint_alnum/train", transform=None)

    datasets = [iiit_train_dataset, realprint_alnum_train_dataset]

    # 可选：保留旧版 realprint
    if os.path.exists("data_realprint/train"):
        old_realprint_train_dataset = OCRDataset("data_realprint/train", transform=None)
        datasets.append(old_realprint_train_dataset)
        print("old realprint train samples:", len(old_realprint_train_dataset))

    train_dataset = ConcatDataset(datasets)

    # 验证集只看 IIIT5K
    val_dataset = OCRDataset("iiit5k_alnum/val", transform=None)

    print("iiit train samples:", len(iiit_train_dataset))
    print("realprint alnum train samples:", len(realprint_alnum_train_dataset))
    print("total train samples:", len(train_dataset))
    print("val samples:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=make_collate_fn(is_train=True)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=make_collate_fn(is_train=False)
    )

    num_classes = len(CHARS) + 1
    model = CRNN(num_classes=num_classes).to(device)

    load_pretrained_partial(model, device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    epochs = 40
    patience = 8
    no_improve_count = 0

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_model_state = None

    train_losses = []
    val_losses = []
    val_accs = []

    for epoch in range(epochs):
        print(f"\n===== Epoch {epoch + 1}/{epochs} =====")
        print(f"current lr: {optimizer.param_groups[0]['lr']:.6f}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_word_acc={val_acc:.4f}"
        )

        improved = False

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_WEIGHTS_LOSS)
            print(f"best model saved by val_loss: {SAVE_WEIGHTS_LOSS}")
            improved = True

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), SAVE_WEIGHTS_ACC)
            print(f"best model saved by val_acc: {SAVE_WEIGHTS_ACC}")
            improved = True

        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
            print(f"no improvement count: {no_improve_count}/{patience}")

        scheduler.step()

        if no_improve_count >= patience:
            print("Early stopping triggered.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    with open("curve.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_accs": val_accs
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print("training finished.")
    print(f"best val_loss: {best_val_loss:.4f}")
    print(f"best val_acc: {best_val_acc:.4f}")
    print("curve saved to: curve.json")


if __name__ == "__main__":
    if not os.path.exists("iiit5k_alnum/train") or not os.path.exists("iiit5k_alnum/val"):
        raise FileNotFoundError("Please run prepare_iiit5k.py first.")
    if not os.path.exists("data_realprint_alnum/train"):
        raise FileNotFoundError("Please run generate_realprint_alnum_dataset.py first.")
    main()