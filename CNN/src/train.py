# src/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import DefectResNet

# ---------------- CONFIG ----------------
DATASET_ROOT = "../dataset"
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- TRANSFORMS ----------------
train_tfms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

val_tfms = transforms.Compose([
    transforms.ToTensor(),
])

# ---------------- DATASETS ----------------
train_ds = datasets.ImageFolder(
    root=f"{DATASET_ROOT}/train",
    transform=train_tfms
)

val_ds = datasets.ImageFolder(
    root=f"{DATASET_ROOT}/test",
    transform=val_tfms
)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True
)

val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False
)

print("Class mapping:", train_ds.class_to_idx)

# ---------------- MODEL ----------------
model = DefectResNet(num_classes=2).to(DEVICE)

# Freeze most layers (IMPORTANT)
for param in model.parameters():
    param.requires_grad = False

for param in model.backbone.layer4.parameters():
    param.requires_grad = True

for param in model.backbone.fc.parameters():
    param.requires_grad = True

# ---------------- TRAINING SETUP ----------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

# ---------------- TRAIN LOOP ----------------
for epoch in range(EPOCHS):
    model.train()
    train_correct = 0
    train_total = 0
    train_loss = 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = outputs.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_acc = train_correct / train_total

    # ---------------- VALIDATION ----------------
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

# ---------------- SAVE MODEL ----------------
torch.save(model.state_dict(), "../models/model.pth")
print("Model saved to models/model.pth")
