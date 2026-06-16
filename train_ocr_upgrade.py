import os
import copy
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image, ImageFilter

from dataset import OCRDataset
from model import CRNN
from utils import encode_text, decode_ctc, CHARS

IMG_H = 48
MIN_W = 48
MAX_W = 256

BATCH_SIZE = 16

PHASE1_EPOCHS = 25
PHASE2_EPOCHS = 15
PATIENCE = 6

SAVE_WEIGHTS_PHASE1_ACC = "best_phase1_acc.pth"
SAVE_WEIGHTS_PHASE1_LOSS = "best_phase1_loss.pth"
SAVE_WEIGHTS_PHASE2_ACC = "best_phase2_acc.pth"
SAVE_WEIGHTS_PHASE2_LOSS = "best_phase2_loss.pth"

def resize_keep_ratio(image, img_h=IMG_H, min_w=MIN_W, max_w=MAX_W):
    w, h = image.size
    new_w = max(1, int(w * img_h / h))
    new_w = max(min_w, min(new_w, max_w))
    image = image.resize((new_w, img_h), Image.BICUBIC)
    return image

def augment_image(image):
    if random.random() < 0.7:
        angle = random.uniform(-4, 4)
        image = image.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)

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

    if random.random() < 0.25:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))

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

            tensor = to_tensor(image)
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

def sample_weight_from_text(text: str) -> float:
    """
    越难的样本权重越高：
    - 长词
    - 含数字
    - 含大写
    """
    w = 1.0
    length = len(text)

    if length >= 8:
        w += 0.8
    if length >= 12:
        w += 0.8

    if any(c.isdigit() for c in text):
        w += 0.8

    if any(c.isupper() for c in text):
        w += 0.5

    return w

def build_weighted_sampler_from_concat_dataset(concat_dataset: ConcatDataset):
    weights = []

    for dataset in concat_dataset.datasets:

        for _, text in dataset.samples:
            weights.append(sample_weight_from_text(text))

    weights = torch.tensor(weights, dtype=torch.double)
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    return sampler

def build_weighted_sampler_from_dataset(dataset: OCRDataset):
    weights = []
    for _, text in dataset.samples:
        weights.append(sample_weight_from_text(text))

    weights = torch.tensor(weights, dtype=torch.double)
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    return sampler

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
        "best_crnn_iiit5k_mix2_acc.pth",
        "best_crnn_iiit5k_mix2_loss.pth",
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

def run_training_stage(
    stage_name,
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    save_acc_path,
    save_loss_path,
    patience=PATIENCE
):
    print(f"\n{'=' * 20} {stage_name} START {'=' * 20}")

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_model_state = None
    no_improve_count = 0

    for epoch in range(epochs):
        print(f"\n===== {stage_name} Epoch {epoch + 1}/{epochs} =====")
        print(f"current lr: {optimizer.param_groups[0]['lr']:.6f}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        print(
            f"{stage_name} Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_word_acc={val_acc:.4f}"
        )

        improved = False

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_loss_path)
            print(f"best model saved by val_loss: {save_loss_path}")
            improved = True

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_acc_path)
            print(f"best model saved by val_acc: {save_acc_path}")
            improved = True

        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
            print(f"no improvement count: {no_improve_count}/{patience}")

        scheduler.step()

        if no_improve_count >= patience:
            print(f"{stage_name}: Early stopping triggered.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\n{stage_name} finished.")
    print(f"best val_loss: {best_val_loss:.4f}")
    print(f"best val_acc: {best_val_acc:.4f}")

    return model, best_val_acc, best_val_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    if not os.path.exists("iiit5k_alnum/train") or not os.path.exists("iiit5k_alnum/val"):
        raise FileNotFoundError("Please prepare iiit5k_alnum first.")

    if not os.path.exists("data_realprint_alnum/train"):
        raise FileNotFoundError("Please generate data_realprint_alnum first.")

    iiit_train_dataset = OCRDataset("iiit5k_alnum/train", transform=None)
    iiit_val_dataset = OCRDataset("iiit5k_alnum/val", transform=None)

    realprint_alnum_train_dataset = OCRDataset("data_realprint_alnum/train", transform=None)

    phase1_datasets = [iiit_train_dataset, realprint_alnum_train_dataset]

    if os.path.exists("data_realprint/train"):
        old_realprint_train_dataset = OCRDataset("data_realprint/train", transform=None)
        phase1_datasets.append(old_realprint_train_dataset)
        print("old realprint train samples:", len(old_realprint_train_dataset))

    phase1_train_dataset = ConcatDataset(phase1_datasets)

    print("iiit train samples:", len(iiit_train_dataset))
    print("realprint alnum train samples:", len(realprint_alnum_train_dataset))
    print("phase1 total train samples:", len(phase1_train_dataset))
    print("val samples:", len(iiit_val_dataset))

    phase1_sampler = build_weighted_sampler_from_concat_dataset(phase1_train_dataset)
    phase2_sampler = build_weighted_sampler_from_dataset(iiit_train_dataset)

    phase1_train_loader = DataLoader(
        phase1_train_dataset,
        batch_size=BATCH_SIZE,
        sampler=phase1_sampler,
        collate_fn=make_collate_fn(is_train=True)
    )

    phase2_train_loader = DataLoader(
        iiit_train_dataset,
        batch_size=BATCH_SIZE,
        sampler=phase2_sampler,
        collate_fn=make_collate_fn(is_train=True)
    )

    val_loader = DataLoader(
        iiit_val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=make_collate_fn(is_train=False)
    )

    num_classes = len(CHARS) + 1
    model = CRNN(num_classes=num_classes).to(device)

    load_pretrained_partial(model, device)

    model, _, _ = run_training_stage(
        stage_name="PHASE1_MIXED",
        model=model,
        train_loader=phase1_train_loader,
        val_loader=val_loader,
        device=device,
        epochs=PHASE1_EPOCHS,
        lr=2e-4,
        save_acc_path=SAVE_WEIGHTS_PHASE1_ACC,
        save_loss_path=SAVE_WEIGHTS_PHASE1_LOSS,
        patience=PATIENCE
    )

    model, _, _ = run_training_stage(
        stage_name="PHASE2_IIIT_FINETUNE",
        model=model,
        train_loader=phase2_train_loader,
        val_loader=val_loader,
        device=device,
        epochs=PHASE2_EPOCHS,
        lr=1e-4,
        save_acc_path=SAVE_WEIGHTS_PHASE2_ACC,
        save_loss_path=SAVE_WEIGHTS_PHASE2_LOSS,
        patience=PATIENCE
    )

    print("\nAll training finished.")
    print("Recommended final weights:")
    print(f"- {SAVE_WEIGHTS_PHASE2_ACC}")
    print(f"- {SAVE_WEIGHTS_PHASE2_LOSS}")

if __name__ == "__main__":
    main()
