import os
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import OCRDataset
from model import CRNN
from utils import encode_text, decode_ctc, CHARS


SAVE_WEIGHTS_ACC = "best_fast_acc.pth"
SAVE_WEIGHTS_LOSS = "best_fast_loss.pth"
CURVE_JSON = "curve_fast.json"


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

        for pred, gt in zip(preds, texts):
            if pred == gt:
                correct += 1
            total += 1

    avg_loss = total_loss / len(loader)
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


def load_pretrained(model, device):
    candidates = [
        "best_fast_acc.pth",
        "best_fast_loss.pth",
        "best_crnn_iiit5k_alnum_acc.pth",
        "best_crnn_iiit5k_alnum_loss.pth",
        "best_crnn_iiit5k_mix_acc.pth",
        "best_crnn_iiit5k_mix_loss.pth",
        "best_crnn_realprint.pth",
        "best_crnn_easy.pth",
    ]

    for path in candidates:
        if os.path.exists(path):
            try:
                model.load_state_dict(torch.load(path, map_location=device))
                print(f"loaded pretrained weights: {path}")
                return
            except Exception:
                continue

    print("[WARN] no pretrained weights loaded, training from scratch.")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_transform = transforms.Compose([
        transforms.Resize((48, 120)),
        transforms.RandomRotation(2),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((48, 120)),
        transforms.ToTensor(),
    ])

    train_dataset = OCRDataset("iiit5k_alnum/train", transform=train_transform)
    val_dataset = OCRDataset("iiit5k_alnum/val", transform=val_transform)

    print("train samples:", len(train_dataset))
    print("val samples:", len(val_dataset))

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

    num_classes = len(CHARS) + 1
    model = CRNN(num_classes=num_classes).to(device)

    load_pretrained(model, device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    epochs = 12
    patience = 4
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

    with open(CURVE_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accs": val_accs
        }, f, ensure_ascii=False, indent=2)

    print("training finished.")
    print(f"best val_loss: {best_val_loss:.4f}")
    print(f"best val_acc: {best_val_acc:.4f}")
    print(f"curve saved to: {CURVE_JSON}")


if __name__ == "__main__":
    if not os.path.exists("iiit5k_alnum/train") or not os.path.exists("iiit5k_alnum/val"):
        raise FileNotFoundError("Please run prepare_iiit5k.py first.")
    main()