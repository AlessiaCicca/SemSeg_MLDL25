import os
import zipfile
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler  
import gdown
import numpy as np
from models.bisenet.build_bisenet import BiSeNet
import datasets.cityscapes as cityscapes

#TRAIN 2B

scaler = GradScaler()  


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

    if len(ious) == 0:
        return 0.0
    return np.mean(ious)

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
    
    '''

    Preliminary setup:
     This initial part of the main function handles dataset retrieval and path construction.
     If the dataset is not already present, it is downloaded and extracted.
     The function then searches for the relevant image and mask directories,
     builds the required paths for training and validation sets,
     and generates CSV files containing image-label pairs for both sets.

    '''

    zip_path = 'cityscapes_dataset.zip'
    base_extract_path = './Cityscapes'

    if not os.path.exists(base_extract_path):
        os.system(f"gdown https://drive.google.com/uc?id=1Qb4UrNsjvlU-wEsR9d7rckB0YS_LXgb2 -O {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(base_extract_path)

    images_dir = find_folder(base_extract_path, 'images')
    masks_dir = find_folder(base_extract_path, 'gtFine')

    if not images_dir or not masks_dir:
        raise RuntimeError("'images' o 'gtFine' not found!")

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

    print(f"\n>>>Training con lr={lr}, batch_size={bs}")
    
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True,
                              num_workers=6, pin_memory=True)  
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False,
                            num_workers=6, pin_memory=True)

    model = BiSeNet(num_classes=19, context_path='resnet18').to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    best_acc = 0
    best_miou = 0

    '''
    Main training and validation loop:
     This loop iterates over all epochs. For each epoch, it performs a training pass
     followed by validation. After validation, it checks the current mIoU value and 
     tracks the best mIoU achieved so far. If a new best mIoU is found, the model 
     checkpoint is saved. Additionally, periodic checkpoints are saved at fixed intervals.

    '''

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
            print(f"Best model -> mIoU: {best_miou:.2f}% (Acc: {val_acc:.2f}%)")

    torch.save(model.state_dict(), f'checkpoints_bisenet/bisenetfinal_model_lr{lr}_bs{bs}.pth')
    print(f"TRAINING COMPLETED  lr={lr}, bs={bs} | Best mIoU: {best_miou:.2f}%")
