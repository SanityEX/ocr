import os
import copy
import json
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model_attention_ocr import AttentionOCR
from utils import (
    VOCAB_SIZE,
    PAD_IDX, SOS_IDX, EOS_IDX,
    encode_text_attention, decode_attention
)

IMG_H = 48
IMG_W = 192
BATCH_SIZE = 20
MAX_LABEL_LEN = 25

PHASE1_EPOCHS = 18
PHASE2_EPOCHS = 25
PATIENCE = 6

MERGE_TRAIN_TXT = r"D:\mnist_project\ocr1\merge_train_v2.txt"

IIIT_TRAIN_TXT = r"D:\mnist_project\ocr1\iiit5k_alnum\train\labels.txt"
IIIT_TRAIN_DIR = r"D:\mnist_project\ocr1\iiit5k_alnum\train\images"

VAL_LABEL_TXT = r"D:\mnist_project\ocr1\iiit5k_alnum\val\labels.txt"
VAL_IMG_DIR = r"D:\mnist_project\ocr1\iiit5k_alnum\val\images"

SAVE_PHASE1_ACC = "best_attention_v2_phase1_acc.pth"
SAVE_PHASE1_LOSS = "best_attention_v2_phase1_loss.pth"
SAVE_PHASE2_ACC = "best_attention_v2_phase2_acc.pth"
SAVE_PHASE2_LOSS = "best_attention_v2_phase2_loss.pth"

CURVE_PHASE1 = "curve_attention_merge_phase1.json"
CURVE_PHASE2 = "curve_attention_merge_phase2.json"

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

def train_transform_medium():
    return transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.RandomRotation(3),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.03, 0.03),
            scale=(0.95, 1.05),
            shear=2
        ),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
    ])

def train_transform_light():
    return transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.RandomRotation(2),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
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

    seqs = [encode_text_attention(t, max_len=MAX_LABEL_LEN) for t in texts]
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
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    sample_batches = set(random.sample(range(len(loader)), k=min(2, len(loader))))

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

        pred_tokens = model.predict(images, SOS_IDX, EOS_IDX, max_len=MAX_LABEL_LEN)
        preds = decode_batch(pred_tokens)

        for i, (pred, gt) in enumerate(zip(preds, texts)):
            pred = pred.strip().upper()
            gt = gt.strip().upper()

            if batch_idx in sample_batches and i < 2:
                tag = "OK" if pred == gt else "NG"
                print(f"[sample batch {batch_idx}] {tag} | GT: {gt} | Pred: {pred}")

            if pred == gt:
                correct += 1
            total += 1

    return total_loss / len(loader), correct / total if total > 0 else 0.0

def run_stage(
    stage_name,
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    save_acc,
    save_loss,
    curve_path,
    label_smoothing,
):
    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_IDX,
        label_smoothing=label_smoothing
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=8e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.5)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_model_state = None
    no_improve = 0

    train_losses = []
    val_losses = []
    val_accs = []

    for epoch in range(epochs):
        print(f"\n===== {stage_name} Epoch {epoch + 1}/{epochs} =====")
        print(f"current lr: {optimizer.param_groups[0]['lr']:.6f}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"{stage_name} Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_word_acc={val_acc:.4f}"
        )

        improved = False

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_loss)
            print(f"best model saved by val_loss: {save_loss}")
            improved = True

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_acc)
            print(f"best model saved by val_acc: {save_acc}")
            improved = True

        if improved:
            no_improve = 0
        else:
            no_improve += 1
            print(f"no improvement count: {no_improve}/{PATIENCE}")

        scheduler.step()

        if no_improve >= PATIENCE:
            print(f"{stage_name}: Early stopping triggered.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    with open(curve_path, "w", encoding="utf-8") as f:
        json.dump({
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accs": val_accs
        }, f, ensure_ascii=False, indent=2)

    print(f"{stage_name} finished.")
    print(f"best val_loss: {best_val_loss:.4f}")
    print(f"best val_acc: {best_val_acc:.4f}")

    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    merge_dataset = TxtOCRDataset(
        MERGE_TRAIN_TXT,
        transform=train_transform_medium()
    )

    iiit_train_dataset = TxtOCRDataset(
        IIIT_TRAIN_TXT,
        transform=train_transform_light(),
        base_dir=IIIT_TRAIN_DIR
    )

    val_dataset = TxtOCRDataset(
        VAL_LABEL_TXT,
        transform=val_transform(),
        base_dir=VAL_IMG_DIR
    )

    if len(merge_dataset) == 0:
        raise ValueError("merge_dataset is empty. Check MERGE_TRAIN_TXT paths.")
    if len(iiit_train_dataset) == 0:
        raise ValueError("iiit_train_dataset is empty. Check IIIT_TRAIN_DIR.")
    if len(val_dataset) == 0:
        raise ValueError("val_dataset is empty. Check VAL_IMG_DIR.")

    train_loader_phase1 = DataLoader(
        merge_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    train_loader_phase2 = DataLoader(
        iiit_train_dataset,
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

    print("phase1 train samples:", len(merge_dataset))
    print("phase2 iiit samples:", len(iiit_train_dataset))
    print("val samples:", len(val_dataset))

    model = AttentionOCR(vocab_size=VOCAB_SIZE).to(device)

    model = run_stage(
        stage_name="PHASE1_MJSYNTH_IIIT",
        model=model,
        train_loader=train_loader_phase1,
        val_loader=val_loader,
        device=device,
        epochs=PHASE1_EPOCHS,
        lr=2e-4,
        save_acc=SAVE_PHASE1_ACC,
        save_loss=SAVE_PHASE1_LOSS,
        curve_path=CURVE_PHASE1,
        label_smoothing=0.05,
    )

    model = run_stage(
        stage_name="PHASE2_IIIT_FINETUNE",
        model=model,
        train_loader=train_loader_phase2,
        val_loader=val_loader,
        device=device,
        epochs=PHASE2_EPOCHS,
        lr=4e-5,
        save_acc=SAVE_PHASE2_ACC,
        save_loss=SAVE_PHASE2_LOSS,
        curve_path=CURVE_PHASE2,
        label_smoothing=0.03,
    )

    print("\nAll training finished.")
    print("Recommended final weights:")
    print(SAVE_PHASE2_ACC)
    print(SAVE_PHASE2_LOSS)

if __name__ == "__main__":
    main()
