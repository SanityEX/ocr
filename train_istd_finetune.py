import os
import copy
import json
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model_attention_ocr import AttentionOCR
from utils import (
    VOCAB_SIZE,
    PAD_IDX,
    SOS_IDX,
    EOS_IDX,
    encode_text_attention,
    decode_attention
)

TRAIN_TXT = r"D:\mnist_project\ocr1\istd_train_10_20_30.txt"

VAL_LABEL_TXT = r"D:\mnist_project\ocr1\word_occluded_10k\labels.txt"
VAL_IMG_DIR = r"D:\mnist_project\ocr1\word_occluded_10k\images"

INIT_WEIGHTS = r"D:\mnist_project\ocr1\best_attention_v2_phase2_acc.pth"

SAVE_ACC = "best_attention_istd_occlusion_acc.pth"
SAVE_LOSS = "best_attention_istd_occlusion_loss.pth"
CURVE_PATH = "curve_attention_istd_occlusion.json"

IMG_H = 48
IMG_W = 192
BATCH_SIZE = 16
MAX_LABEL_LEN = 25

EPOCHS = 8
PATIENCE = 3
LR = 5e-6

class TxtOCRDataset(Dataset):
    def __init__(self, txt_path, transform=None, base_dir=None):
        self.samples = []
        self.transform = transform
        self.base_dir = base_dir

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

                if base_dir is not None:
                    path = os.path.join(base_dir, os.path.basename(path))

                path = path.replace("\\", "/")

                if os.path.exists(path):
                    self.samples.append((path, label))

        print(f"loaded {len(self.samples)} samples from {txt_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = Image.open(path).convert("L")

        if self.transform:
            img = self.transform(img)

        return img, label

def train_transform():
    return transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.RandomRotation(2),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.02, 0.02),
            scale=(0.97, 1.03),
            shear=1
        ),
        transforms.ToTensor(),
    ])

def val_transform():
    return transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ToTensor(),
    ])

def collate_fn(batch):
    images, texts = zip(*batch)
    images = torch.stack(images, dim=0)

    seqs = [
        encode_text_attention(t, max_len=MAX_LABEL_LEN)
        for t in texts
    ]

    max_len = max(len(s) for s in seqs)
    padded = []

    for s in seqs:
        s = s + [PAD_IDX] * (max_len - len(s))
        padded.append(s)

    target = torch.tensor(padded, dtype=torch.long)

    return images, texts, target

def clean_decode(seq):
    result = []

    for idx in seq:
        if idx == EOS_IDX:
            break

        if idx >= 3:
            result.append(idx)

    return decode_attention(result)

@torch.no_grad()
def decode_batch(pred_tokens):
    results = []

    for seq in pred_tokens.cpu().tolist():
        results.append(clean_decode(seq))

    return results

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, texts, target) in enumerate(loader):
        if batch_idx % 50 == 0:
            print(f"train batch: {batch_idx}/{len(loader)}")

        images = images.to(device)
        target = target.to(device)

        decoder_input = target[:, :-1]
        decoder_target = target[:, 1:]

        logits = model(images, decoder_input)

        B, L, V = logits.size()

        loss = criterion(
            logits.reshape(B * L, V),
            decoder_target.reshape(B * L)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, texts, target) in enumerate(loader):
        if batch_idx % 20 == 0:
            print(f"val batch: {batch_idx}/{len(loader)}")

        images = images.to(device)
        target = target.to(device)

        decoder_input = target[:, :-1]
        decoder_target = target[:, 1:]

        logits = model(images, decoder_input)

        B, L, V = logits.size()

        loss = criterion(
            logits.reshape(B * L, V),
            decoder_target.reshape(B * L)
        )

        total_loss += loss.item()

        pred_tokens = model.predict(
            images,
            SOS_IDX,
            EOS_IDX,
            max_len=MAX_LABEL_LEN
        )

        preds = decode_batch(pred_tokens)

        for pred, gt in zip(preds, texts):
            if pred.strip().upper() == gt.strip().upper():
                correct += 1
            total += 1

    return total_loss / len(loader), correct / total if total > 0 else 0.0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_dataset = TxtOCRDataset(
        TRAIN_TXT,
        transform=train_transform()
    )

    val_dataset = TxtOCRDataset(
        VAL_LABEL_TXT,
        transform=val_transform(),
        base_dir=VAL_IMG_DIR
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    model = AttentionOCR(vocab_size=VOCAB_SIZE).to(device)

    model.load_state_dict(
        torch.load(
            INIT_WEIGHTS,
            map_location=device
        )
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_IDX,
        label_smoothing=0.02
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=5e-5
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.5
    )

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    train_losses = []
    val_losses = []
    val_accs = []

    for epoch in range(EPOCHS):
        print(f"\n===== ISTD Fine-tune Epoch {epoch + 1}/{EPOCHS} =====")
        print(f"lr: {optimizer.param_groups[0]['lr']:.8f}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"epoch {epoch + 1} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

        improved = False

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_LOSS)
            print(f"saved best loss model: {SAVE_LOSS}")
            improved = True

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), SAVE_ACC)
            print(f"saved best acc model: {SAVE_ACC}")
            improved = True

        if improved:
            no_improve = 0
        else:
            no_improve += 1
            print(f"no improvement: {no_improve}/{PATIENCE}")

        scheduler.step()

        if no_improve >= PATIENCE:
            print("Early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    with open(CURVE_PATH, "w", encoding="utf-8") as f:
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

    print("\nISTD Fine-tune finished.")
    print("Best val acc:", best_val_acc)
    print("Recommended weights:", SAVE_ACC)

if __name__ == "__main__":
    main()
