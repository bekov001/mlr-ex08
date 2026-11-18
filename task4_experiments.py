"""Task 4: a better leaf classifier. One change at a time against a fixed baseline.

Starts from the Task 3 pipeline (LeafDataset, LeafNet, the same training loop) and adds
only what each variant needs. Every variant: the same training images, seeds 0, 1, 2,
measured on the validation set and on a camera-like validation set generated once.
"""
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import models, transforms

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
SEEDS = [0, 1, 2]


# ---- Task 3 pipeline (unchanged) -------------------------------------------------------
class LeafDataset(Dataset):
    def __init__(self, folder, transform):
        self.transform = transform
        self.classes = sorted(d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)))
        self.samples = []
        for index, name in enumerate(self.classes):
            class_folder = os.path.join(folder, name)
            for file in sorted(os.listdir(class_folder)):
                self.samples.append((os.path.join(class_folder, file), index))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return self.transform(Image.open(path).convert("RGB")), label


class LeafNet(nn.Module):
    def __init__(self, batch_norm=False, dropout=0.0):
        super().__init__()
        channels = [3, 16, 32, 64, 64]
        self.convs = nn.ModuleList(nn.Conv2d(channels[i], channels[i + 1], 3, padding=1) for i in range(4))
        # Variant "batch norm": one BatchNorm2d after every convolution.
        self.bns = nn.ModuleList(nn.BatchNorm2d(channels[i + 1]) if batch_norm else nn.Identity() for i in range(4))
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 6 * 6, 64)
        self.drop = nn.Dropout(dropout)  # variant "dropout"; p=0 does nothing
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        for conv, bn in zip(self.convs, self.bns):
            x = self.pool(torch.relu(bn(conv(x))))
        x = self.drop(torch.relu(self.fc1(x.flatten(1))))
        return self.fc2(x)


# ---- data --------------------------------------------------------------------------------
plain = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
geometric = [
    transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
]
colour = [transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)]
aug_geo = transforms.Compose(geometric + [transforms.ToTensor(), transforms.Normalize(mean, std)])
aug_geo_colour = transforms.Compose(geometric + colour + [transforms.ToTensor(), transforms.Normalize(mean, std)])

val_ds = LeafDataset("data/leaves/val", plain)
val_loader = DataLoader(val_ds, batch_size=64)


def camera_like_val():
    """The validation images randomly cropped, rotated and re-coloured — generated ONCE, seed 1234."""
    camera = transforms.Compose([
        transforms.RandomResizedCrop(96, scale=(0.5, 0.9), ratio=(0.8, 1.25)),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05),
        transforms.GaussianBlur(3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    torch.manual_seed(1234)
    raw = LeafDataset("data/leaves/val", camera)
    xs, ys = zip(*(raw[i] for i in range(len(raw))))
    return DataLoader(TensorDataset(torch.stack(xs), torch.tensor(ys)), batch_size=64)


camera_loader = camera_like_val()


def accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(dim=1) == labels).sum().item()
            total += len(labels)
    return correct / total


# ---- models ------------------------------------------------------------------------------
def mobilenet(freeze=False):
    model = models.mobilenet_v3_small(weights="DEFAULT")
    if freeze:  # variant "transfer, frozen": only the new last layer learns
        for p in model.parameters():
            p.requires_grad = False
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)
    return model


# ---- the training loop of Task 3 (+ optional scheduler / label smoothing / best epoch) --
def train(make_model, train_transform, seed, epochs=15, lr=1e-3, cosine=False, label_smoothing=0.0,
          weight_decay=0.0, keep_best=False):
    torch.manual_seed(seed)
    model = make_model().to(device)
    train_loader = DataLoader(LeafDataset("data/leaves/train", train_transform), batch_size=32, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = (torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay) if weight_decay
                 else torch.optim.Adam(params, lr=lr))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs) if cosine else None
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    best_acc, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        if scheduler:
            scheduler.step()
        if keep_best:
            acc = accuracy(model, val_loader)
            if acc > best_acc:
                best_acc, best_state = acc, copy.deepcopy(model.state_dict())
    if keep_best:
        model.load_state_dict(best_state)
    return model


VARIANTS = {
    # name: (description, kwargs)
    "A baseline": ("Task 3 LeafNet, no augmentation, Adam 1e-3, 15 ep", dict(make_model=LeafNet, train_transform=plain)),
    "B +geometric aug": ("A + crop/flip/rotation", dict(make_model=LeafNet, train_transform=aug_geo)),
    "C +batch norm": ("A + BatchNorm2d", dict(make_model=lambda: LeafNet(batch_norm=True), train_transform=plain)),
    "D aug+BN": ("A + geometric aug + BN (Task 3 result)", dict(make_model=lambda: LeafNet(batch_norm=True), train_transform=aug_geo)),
    "E D+colour jitter": ("D + ColorJitter", dict(make_model=lambda: LeafNet(batch_norm=True), train_transform=aug_geo_colour)),
    "F D+dropout+AdamW+LS, 30 ep cosine": ("D + dropout 0.3, AdamW wd 1e-4, label smoothing 0.1, cosine LR, 30 ep, best epoch",
        dict(make_model=lambda: LeafNet(batch_norm=True, dropout=0.3), train_transform=aug_geo_colour, epochs=30,
             cosine=True, label_smoothing=0.1, weight_decay=1e-4, keep_best=True)),
    "G MobileNet frozen": ("pretrained MobileNetV3-S, only the last layer trained", dict(make_model=lambda: mobilenet(freeze=True), train_transform=plain)),
    "H MobileNet fine-tune": ("pretrained MobileNetV3-S, whole network (= Task 2 model)", dict(make_model=mobilenet, train_transform=plain)),
    "I H+aug+colour, cosine 25 ep": ("H + geometric + colour aug, cosine LR, 25 ep",
        dict(make_model=mobilenet, train_transform=aug_geo_colour, epochs=25, cosine=True)),
}

if __name__ == "__main__":
    only = sys.argv[1:]
    results = {}
    for name, (desc, kwargs) in VARIANTS.items():
        if only and not any(name.startswith(o) for o in only):
            continue
        vals, cams = [], []
        t0 = time.time()
        for seed in SEEDS:
            model = train(seed=seed, **kwargs)
            vals.append(accuracy(model, val_loader))
            cams.append(accuracy(model, camera_loader))
            if name.startswith("I") and seed == 0:
                torch.save(model.state_dict(), "best_mobilenet_aug.pt")
        results[name] = dict(desc=desc, val=vals, camera=cams)
        print(f"{name:38s} val {np.mean(vals):.3f} ± {np.std(vals):.3f}  camera {np.mean(cams):.3f} ± {np.std(cams):.3f}"
              f"  ({time.time() - t0:.0f} s)", flush=True)
    json.dump(results, open("task4_results.json", "w"), indent=1)
