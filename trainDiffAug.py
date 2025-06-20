import os
import zipfile
import gc
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import gdown
import time
import csv
import sys
from PIL import Image
from tqdm import tqdm
import numpy as np

from models.bisenet.build_bisenet import BiSeNet
import datasets.gta5 as GTA5
from augmentation import CombinedAugmentation, val_transform_fn

scaler = GradScaler()

def compute_class_weights(label_dir, num_classes=19):
    class_pixel_counts = np.zeros(num_classes, dtype=np.int64)
    mask_paths = glob(os.path.join(label_dir, "*.png"))
    for mask_path in tqdm(mask_paths, desc="Calcolo frequenze classi"):
        mask = np.array(Image.open(mask_path))
        for class_id in range(num_classes):
            class_pixel_counts[class_id] += np.sum(mask == class_id)
    total_pixels = np.sum(class_pixel_counts)
    class_freqs = class_pixel_counts / total_pixels
    weights = 1.0 / (np.log(1.02 + class_freqs))
    return torch.FloatTensor(weights)

def compute_miou(preds, labels, num_classes=19, ignore_index=255):
    ious = []
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_inds = (preds == cls)
        label_inds = (labels == cls)
        intersection = np.logical_and(pred_inds, label_inds).sum()
        union = np.logical_or(pred_inds, label_inds).sum()
        if union == 0:
            continue
        iou = intersection / union
        ious.append(iou)
    return np.mean(ious) if ious else 0.0

def train(epoch, model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs[0], targets.squeeze(1).long())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        _, predicted = torch.max(outputs[0], 1)
        correct += (predicted == targets.squeeze(1)).sum().item()
        total += torch.numel(targets.squeeze(1))
        del inputs, targets, outputs, predicted, loss
        gc.collect()
    acc = 100. * correct / total
    print(f'Train Epoch {epoch} - Loss: {running_loss / len(train_loader):.4f} - Acc: {acc:.2f}%')

def validate(model, val_loader, criterion, device, num_classes=19):
    model.eval()
    val_loss, correct, total, miou_total, count = 0, 0, 0, 0, 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets.squeeze(1).long())
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets.squeeze(1)).sum().item()
            total += torch.numel(targets.squeeze(1))
            miou_total += compute_miou(predicted, targets.squeeze(1), num_classes=num_classes)
            count += 1
            del inputs, targets, outputs, predicted, loss
            gc.collect()
    acc = 100. * correct / total
    mean_iou = 100. * (miou_total / count) if count > 0 else 0
    print(f'Validation - Loss: {val_loss / len(val_loader):.4f} - Acc: {acc:.2f}% - mIoU: {mean_iou:.2f}%')
    return acc, mean_iou

def find_folder(start_path, folder_name):
    for root, dirs, _ in os.walk(start_path):
        if folder_name in dirs:
            return os.path.join(root, folder_name)
    return None

if _name_ == "_main_":
    print(">>>Training...")
    tipo = 1
    base_extract_path = './tmp/GTA5'
    zip_path = 'gt5_dataset.zip'
    gdrive_id = "1xYxlcMR2WFCpayNrW2-Rb7N-950vvl23"
    gdown_url = f"https://drive.google.com/uc?id={gdrive_id}"

    if not os.path.exists(base_extract_path):
        print("Download dataset...")
        gdown.download(gdown_url, zip_path, quiet=False)
        if zipfile.is_zipfile(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(base_extract_path)
        else:
            print("Error.")
            os.remove(zip_path)
    else:
        print(" Dataset already present.")

    train_images_dir = find_folder(base_extract_path, 'images')
    train_masks_dir = find_folder(base_extract_path, 'labels')
    train_csv = 'train_gta5_annotations.csv'
    val_csv = 'val_gta5_annotations.csv'
    GTA5.create_gta5_csv(train_images_dir, train_masks_dir, train_csv, val_csv, base_extract_path)
    preprocessed_masks_dir = './tmp/GTA5/GTA5/labels_trainid'

    base_train_dataset = GTA5.GTA5(train_csv, base_extract_path, transform=None,
                                   target_transform=None, mask_preprocessed_dir=preprocessed_masks_dir)

    if tipo == 1:
        train_transform = CombinedAugmentation(base_train_dataset, use_flip=True, use_colorjitter=True, use_scale=True, use_crop=True, use_classmix=True,
                                               use_brightness=False, use_hue=False, use_gamma=False, use_saturation=False, use_contrast=False) # SET B
    elif tipo == 2:
        train_transform = CombinedAugmentation(base_train_dataset, use_flip=True, use_scale=True, use_crop=True, use_colorjitter=False, use_classmix=False,
                                               use_brightness=True, use_hue=True, use_gamma=True, use_saturation=True, use_contrast=True) # SET A
    elif tipo == 3:
        train_transform = CombinedAugmentation(base_train_dataset, use_flip=True, use_scale=False, use_crop=True, use_colorjitter=False, use_classmix=False,
                                               use_brightness=False, use_hue=False, use_gamma=False, use_saturation=False, use_contrast=False) # CROP / FLIP
    elif tipo == 4:
        train_transform = CombinedAugmentation(base_train_dataset, use_flip=False, use_scale=False, use_crop=False, use_colorjitter=True, use_classmix=False,
                                               use_brightness=False, use_hue=False, use_gamma=False, use_saturation=False, use_contrast=False) # JITTER 
    elif tipo == 5:
        train_transform = CombinedAugmentation(base_train_dataset, use_flip=True, use_scale=False, use_crop=False, use_colorjitter=False, use_classmix=False,
                                               use_brightness=False, use_hue=False, use_gamma=False, use_saturation=False, use_contrast=False) # FLIP 
    else:
        raise ValueError("Tipo non valido. Scegli da 1 a 5.")

    train_dataset = GTA5.GTA5(train_csv, base_extract_path, transform=train_transform,
                              target_transform=None, mask_preprocessed_dir=preprocessed_masks_dir)

    val_dataset = GTA5.GTA5(val_csv, base_extract_path, transform=val_transform_fn,
                            target_transform=None, mask_preprocessed_dir=preprocessed_masks_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs('checkpoints_aug', exist_ok=True)
    os.makedirs('logs_aug', exist_ok=True)

    num_epochs = 50
    lr = 0.01
    bs = 4

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)

    model = BiSeNet(num_classes=19, context_path='resnet18').to(device)
    class_weights = compute_class_weights(preprocessed_masks_dir).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    best_miou = 0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train(epoch, model, train_loader, criterion, optimizer, device)
        val_acc, val_miou = validate(model, val_loader, criterion, device)

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), f'checkpoints_aug/best_model_type_cs2.pth')

    torch.save(model.state_dict(), f'checkpoints_aug/final_model_type_cs2.pth')
