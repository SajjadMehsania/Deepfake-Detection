import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
 #   for filename in filenames:
  #      print(os.path.join(dirname, filename))

import os
import random
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import cv2
DATA_DIR = "/kaggle/input/deepfake-face-images/Final Dataset"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-4
PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Affine(translate_percent=0.1, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
    A.Normalize(),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2()
])
class DeepFakeDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.filepaths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        if self.transform:
            img = self.transform(image=img)['image']
        return img, torch.tensor(label, dtype=torch.long)
real_files = glob(os.path.join(DATA_DIR, "Real", "*"))
fake_files = glob(os.path.join(DATA_DIR, "Fake", "*"))

print("✔️ Real images found:", len(real_files))
print("✔️ Fake images found:", len(fake_files))

all_files = real_files + fake_files
labels = [1]*len(real_files) + [0]*len(fake_files)
#✔️ Real images found: 5890
#✔️ Fake images found: 7000
train_f, val_f, train_l, val_l = train_test_split(
    all_files, labels, stratify=labels, test_size=0.2, random_state=42
)

train_ds = DeepFakeDataset(train_f, train_l, transform=train_transform)
val_ds = DeepFakeDataset(val_f, val_l, transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights)

model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

best_val_loss = float('inf')
patience_counter = 0
PATIENCE = 5


for epoch in range(EPOCHS):
    model.train()
    running_loss, running_corrects = 0.0, 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        running_corrects += (outputs.argmax(1) == labels).sum().item()

    train_loss = running_loss / len(train_ds)
    train_acc = running_corrects / len(train_ds)

    model.eval()
    val_loss, val_corrects = 0.0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * imgs.size(0)
            val_corrects += (outputs.argmax(1) == labels).sum().item()

    val_loss /= len(val_ds)
    val_acc = val_corrects / len(val_ds)
    scheduler.step(val_loss)

    print(f"📊 Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Acc = {train_acc:.4f} | Val Loss = {val_loss:.4f}, Acc = {val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        patience_counter = 0
        print("✅ Saved best model.")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("⏹️ Early stopping.")
            break

model.load_state_dict(torch.load("best_model.pth"))
model.eval()

val_corrects = 0
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        val_corrects += (outputs.argmax(1) == labels).sum().item()

final_val_acc = val_corrects / len(val_ds)
print(f"🏁 Final Validation Accuracy: {final_val_acc:.4f}")
######🏁 Final Validation Accuracy: 0.9934
 
