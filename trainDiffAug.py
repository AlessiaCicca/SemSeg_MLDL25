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
import subprocess
from models.bisenet.build_bisenet import BiSeNet
import datasets.gta5 as GTA5
from augmentation import CombinedAugmentation, val_transform_fn

scaler = GradScaler()

def compute_class_weights(label_dir, num_classes=19):
    """
    Computes class weights based on pixel frequency.

    Args:
        label_dir : Directory containing label masks (PNG files) where each pixel represents a class ID.
        num_classes : Number of classes to consider (default: 19).

    Returns:
        torch.FloatTensor: Computed class weights as a 1D tensor of size (num_classes).

    This function iterates over all mask files in the given directory and counts the number of pixels 
    belonging to each class. It then computes the frequency of each class as the ratio between the 
    number of pixels of that class and the total number of pixels. The class weight is calculated 
    using the formula 1 / log(1.02 + class frequency).
    
    """
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
    """
    Computes the mean Intersection over Union (mIoU) metric.

    Args:
        preds (Tensor): Predicted class indices for each pixel.
        labels (Tensor): Ground truth class indices for each pixel.
        num_classes (int): Number of valid classes (default: 19).
        ignore_index (int): Class index to ignore during evaluation (default: 255).

    Returns:
        float: Mean IoU across all valid classes.
    
    This function converts predictions and labels to NumPy arrays and iterates over each class.
    For each class, it computes the intersection and union of predicted and true pixels.
    The IoU for a class is defined as the ratio of intersection over union.
    The final mIoU is the mean of IoUs for all classes present in the evaluation.

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
            continue
        iou = intersection / union
        ious.append(iou)
    return np.mean(ious) if ious else 0.0

def train(epoch, model, train_loader, criterion, optimizer, device):
    """
    Runs a single training epoch.

    Args:
        epoch : Current epoch index (starting from 0).
        model : The model to be trained.
        train_loader : DataLoader providing the training data batches.
        criterion : Loss function used to compute the training loss.
        optimizer : Optimizer used to update model parameters.
        device : The device on which computations are performed (CPU or GPU).

    This function iterates over the training DataLoader, performs forward passes,
    computes the loss, executes backpropagation, updates the model weights,
    and accumulates accuracy metrics over the epoch.

    """
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
    print(f'Train Epoch {epoch+1} - Loss: {running_loss / len(train_loader):.4f} - Acc: {acc:.2f}%')

def validate(model, val_loader, criterion, device, num_classes=19):
    """
    Performs model evaluation on the validation dataset.

    Args:
        model : The model to be evaluated.
        val_loader : DataLoader providing the validation data batches.
        criterion : Loss function used to compute the validation loss.
        device : The device on which computations are performed (CPU or GPU).
        num_classes : Number of valid classes for evaluation (default: 19).

    This function sets the model to evaluation mode and disables gradient computation.
    It iterates over the validation DataLoader, computes predictions, loss,
    and evaluation metrics including accuracy and mean Intersection over Union (mIoU).

    """
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
    """
    Searches for a folder within a directory tree and returns its full path.

    Args:
        start_path : Root directory to start the search.
        folder_name : Name of the target folder to locate.

    Returns:
        str or None: The full path to the folder if found, otherwise None.

    This function recursively walks through the directory tree starting at start_path,
    and returns the full path to the first occurrence of folder_name.

    """
    for root, dirs, _ in os.walk(start_path):
        if folder_name in dirs:
            return os.path.join(root, folder_name)
    return None

if __name__ == "__main__":
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
    result = subprocess.run(['python3', 'preprocess_mask.py'], capture_output=True, text=True)
    print("Output preprocess_mask.py:\n", result.stdout)
    if result.stderr:
        print("Error preprocess_mask.py:\n", result.stderr)
    preprocessed_masks_dir = './tmp/GTA5/GTA5/labels_trainid'

    base_train_dataset = GTA5.GTA5(train_csv, base_extract_path, transform=None,
                                   target_transform=None, mask_preprocessed_dir=preprocessed_masks_dir)
    # ----------------------- #
    #      AUGMENTATIONS      #
    # ----------------------- #

    # Different types of augmentations
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
        raise ValueError("Parameter tipo not valid!")

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
    print(f"New best model saved with mIoU: {best_miou:.2f}% (Acc: {val_acc:.2f}%)")

