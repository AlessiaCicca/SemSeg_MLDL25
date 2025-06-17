import os
import zipfile
import shutil
import gc
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler  
import gdown
import time
from PIL import Image
from tqdm import tqdm
import numpy as np

from models.bisenet.build_bisenet import BiSeNet
import datasets.gta5WithoutRGB as GTA5
import datasets.cityscapes as cityscapes


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

    # Formula del paper ENet (Weighted Cross Entropy)
    weights = 1.0 / (np.log(1.02 + class_freqs))
    return torch.FloatTensor(weights)

def compute_miou(preds, labels, num_classes=19, ignore_index=255):
    """
    Calcola la mean Intersection over Union (mIoU).
    preds: Tensor [N, H, W] - predizioni per pixel (classe)
    labels: Tensor [N, H, W] - ground truth per pixel
    """
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
            # Classe non presente in questo batch
            continue
        iou = intersection / union
        ious.append(iou)

    if len(ious) == 0:
        return 0.0
    return np.mean(ious)

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
    val_loss, correct, total = 0, 0, 0
    miou_total = 0
    count = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets.squeeze(1).long())

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets.squeeze(1)).sum().item()
            total += torch.numel(targets.squeeze(1))

            miou_batch = compute_miou(predicted, targets.squeeze(1), num_classes=num_classes)
            miou_total += miou_batch
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
    print(">>> Avvio training...")

    zip_path = 'cityscapes_dataset.zip'
    base_extract_path = './Cityscapes'

    if not os.path.exists(base_extract_path):
        print(" Dataset non trovato o incompleto, lo scarico...")
        os.system(f"gdown https://drive.google.com/uc?id=1Qb4UrNsjvlU-wEsR9d7rckB0YS_LXgb2 -O {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(base_extract_path)
        print(" Estrazione completata.")
    else:
        print(" Dataset già presente.")

    images_dir = find_folder(base_extract_path, 'images')
    masks_dir = find_folder(base_extract_path, 'gtFine')

    if not images_dir or not masks_dir:
        raise RuntimeError("'images' o 'gtFine' non trovati dopo l’estrazione!")

    train_images_dir = os.path.join(images_dir, 'train')
    val_images_dir = os.path.join(images_dir, 'val')
    train_masks_dir = os.path.join(masks_dir, 'train')
    val_masks_dir = os.path.join(masks_dir, 'val')

    base_path = os.path.commonpath([images_dir, masks_dir])
    train_csv = 'train_annotations.csv'
    val_csv = 'val_annotations.csv'

    cityscapes.create_cityscapes_csv(train_images_dir, train_masks_dir, train_csv, base_path)
    cityscapes.create_cityscapes_csv(val_images_dir, val_masks_dir, val_csv, base_path)

    train_dataset = cityscapes.CityScapes(
        annotations_file=train_csv,
        root_dir=base_path,
        transform=cityscapes.transform['image'],
        target_transform=cityscapes.transform['mask']
    )

    val_dataset = cityscapes.CityScapes(
        annotations_file=val_csv,
        root_dir=base_path,
        transform=cityscapes.transform['image'],
        target_transform=cityscapes.transform['mask']
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs('checkpoints_bisenet', exist_ok=True)

    num_epochs = 50
    lr= 0.01
    bs= 4

    print(f"\n>>> Inizio training con lr={lr}, batch_size={bs}")
    
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True,
                              num_workers=6, pin_memory=True)  
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False,
                            num_workers=6, pin_memory=True)

    model = BiSeNet(num_classes=19, context_path='resnet18').to(device)
    # trainid_mask_dir = "./tmp/GTA5/GTA5/labels_trainid"
    # class_weights = compute_class_weights(trainid_mask_dir).to(device)

    # Crea la loss pesata
    # criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    best_acc = 0
    best_miou = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train(epoch, model, train_loader, criterion, optimizer, device)
        val_acc, val_miou = validate(model, val_loader, criterion, device)

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            torch.save(model.state_dict(), f'checkpoints_bisenet/checkpointBisenet_epoch_{epoch}_lr{lr}_bs{bs}.pth')
            print(f"checkpoint epoch {epoch}")

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), 'checkpoints_bisenet/bisenetbest_model.pth')
            print(f"Nuovo best model con mIoU: {best_miou:.2f}% (Acc: {val_acc:.2f}%)")

    torch.save(model.state_dict(), f'checkpoints_bisenet/bisenetfinal_model_lr{lr}_bs{bs}.pth')
    print(f"Fine training per lr={lr}, bs={bs} | Best mIoU: {best_miou:.2f}%")
