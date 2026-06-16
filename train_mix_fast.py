import os
import json
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms

from dataset import OCRDataset
from model import CRNN
from utils import encode_text, decode_ctc, CHARS


SAVE_ACC = "best_mix_acc.pth"
SAVE_LOSS = "best_mix_loss.pth"
CURVE = "curve_mix.json"


# ===== collate =====
def collate_fn(batch):
    images, texts = zip(*batch)
    images = torch.stack(images, dim=0)

    targets = []
    target_lengths = []

    for text in texts:
        encoded = encode_text(text)
        targets.extend(encoded)
        target_lengths.append(len(encoded))

    targets = torch.tensor(targets, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    return images, texts, targets, target_lengths


# ===== decode =====
@torch.no_grad()
def greedy_decode(logits):
    preds = logits.argmax(dim=2)
    preds = preds.permute(1, 0)

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


# ===== train =====
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for i, (images, texts, targets, target_lengths) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        logits = model(images)
        log_probs = logits.log_softmax(2)

        input_lengths = torch.full(
            (images.size(0),),
            log_probs.size(0),
            dtype=torch.long,
            device=device
        )

        loss = criterion(log_probs, targets, input_lengths, target_lengths)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()

        if i % 20 == 0:
            print(f"train batch: {i}/{len(loader)}")

    return total_loss / len(loader)


# ===== validate =====
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for i, (images, texts, targets, target_lengths) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        logits = model(images)
        log_probs = logits.log_softmax(2)

        input_lengths = torch.full(
            (images.size(0),),
            log_probs.size(0),
            dtype=torch.long,
            device=device
        )

        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        total_loss += loss.item()

        preds = greedy_decode(logits)

        for p, t in zip(preds, texts):
            if p == t:
                correct += 1
            total += 1

        if i % 20 == 0:
            print(f"val batch: {i}/{len(loader)}")

    return total_loss / len(loader), correct / total


# ===== main =====
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    transform = transforms.Compose([
        transforms.Resize((48, 120)),
        transforms.ToTensor()
    ])

    # ===== 数据集 =====
    ds_iiit = OCRDataset("iiit5k_easy/train", transform=transform)

    ds_real = OCRDataset("data_realprint/train", transform=transform)

    ds_real_small = torch.utils.data.Subset(
        ds_real,
        list(range(min(len(ds_real), 1000)))
    )

    train_dataset = ConcatDataset([ds_iiit, ds_real_small])

    val_dataset = OCRDataset("iiit5k_easy/val", transform=transform)

    # 👉 控制realprint比例（避免太多）
    ds_real_small = torch.utils.data.Subset(
        ds_real,
        list(range(min(len(ds_real), 1000)))
    )

    train_dataset = ConcatDataset([ds_iiit, ds_real_small])

    val_dataset = OCRDataset("iiit5k_easy/val", transform=transform)

    print("train size:", len(train_dataset))
    print("val size:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = CRNN(len(CHARS) + 1).to(device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    epochs = 20

    train_losses, val_losses, val_accs = [], [], []

    best_acc = 0
    best_loss = float("inf")

    for epoch in range(epochs):
        print(f"\n===== Epoch {epoch+1}/{epochs} =====")

        tl = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl, va = validate(model, val_loader, criterion, device)

        train_losses.append(tl)
        val_losses.append(vl)
        val_accs.append(va)

        print(f"train_loss={tl:.4f} val_loss={vl:.4f} acc={va:.4f}")

        if va > best_acc:
            best_acc = va
            torch.save(model.state_dict(), SAVE_ACC)

        if vl < best_loss:
            best_loss = vl
            torch.save(model.state_dict(), SAVE_LOSS)

    with open(CURVE, "w") as f:
        json.dump({
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accs": val_accs
        }, f)

    print("done.")


if __name__ == "__main__":
    main()