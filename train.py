import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import OCRDataset
from model import CRNN
from utils import encode_text, decode_ctc, CHARS


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
    # logits: [T, B, C]
    preds = logits.argmax(dim=2)  # [T, B]
    preds = preds.permute(1, 0)   # [B, T]

    results = []
    for seq in preds:
        results.append(decode_ctc(seq.cpu().tolist()))
    return results


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, texts, targets, target_lengths in loader:
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        logits = model(images)              # [T, B, C]
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
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, texts, targets, target_lengths in loader:
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 训练用（带增强）
    train_transform = transforms.Compose([
        transforms.Resize((32, 120)),
        transforms.RandomRotation(degrees=3),
        transforms.ToTensor(),
    ])

    # 验证用（不增强）
    val_transform = transforms.Compose([
        transforms.Resize((32, 120)),
        transforms.ToTensor(),
    ])


    train_dataset = OCRDataset("data/train", transform=train_transform)
    val_dataset = OCRDataset("data/val", transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )

    num_classes = len(CHARS) + 1  # +1 for CTC blank
    model = CRNN(num_classes=num_classes).to(device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10
    best_val_loss = float("inf")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_word_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_crnn.pth")
            print("best model saved: best_crnn.pth")

    print("training finished.")


if __name__ == "__main__":
    if not os.path.exists("data/train") or not os.path.exists("data/val"):
        raise FileNotFoundError("Please make sure data/train and data/val exist.")
    main()