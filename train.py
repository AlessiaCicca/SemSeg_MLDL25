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

from models.deeplabv2.deeplabv2 import get_deeplab_v2
import datasets.cityscapes as cityscapes

scaler = GradScaler()  

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


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets.squeeze(1).long())

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets.squeeze(1)).sum().item()
            total += torch.numel(targets.squeeze(1))

            del inputs, targets, outputs, predicted, loss
            gc.collect()

    acc = 100. * correct / total
    print(f'Validation - Loss: {val_loss / len(val_loader):.4f} - Acc: {acc:.2f}%')
    return acc


def find_folder(start_path, folder_name):
    for root, dirs, _ in os.walk(start_path):
        if folder_name in dirs:
            return os.path.join(root, folder_name)
    return None


if __name__ == "__main__":
    print(">>> Avvio training...")

    zip_path = 'cityscapes_dataset.zip'
    base_extract_path = './Cityscapes'

    """if not os.path.exists(base_extract_path):
        print(" Dataset non trovato o incompleto, lo scarico...")
        os.system(f"gdown https://drive.google.com/uc?id=1Qb4UrNsjvlU-wEsR9d7rckB0YS_LXgb2 -O {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(base_extract_path)
        print(" Estrazione completata.")
    else:
        print(" Dataset già presente.")"""

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
    os.makedirs('checkpoints', exist_ok=True)

    num_epochs = 50
    learning_rates = [0.000025, 0.001]
    batch_sizes = [1, 4, 8]
 

    for lr in learning_rates:
        for bs in batch_sizes:
            print(f"\n>>> Inizio training con lr={lr}, batch_size={bs}")

            train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True,
                                      num_workers=2, pin_memory=True)  
            val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False,
                                    num_workers=2, pin_memory=True)

            model = get_deeplab_v2(num_classes=19).to(device)
            criterion = nn.CrossEntropyLoss(ignore_index=255)
            optimizer = optim.SGD(model.optim_parameters(lr = lr), lr=lr, momentum=0.9, weight_decay=5e-4)

            best_acc = 0
            best_lr=0
            best_bs=0

            for epoch in range(num_epochs):
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                train(epoch, model, train_loader, criterion, optimizer, device)
                val_acc = validate(model, val_loader, criterion, device)

                if epoch % 10 == 0 or epoch == num_epochs - 1:
                    torch.save(model.state_dict(), f'checkpoints/checkpoint_epoch_{epoch}_lr{lr}_bs{bs}.pth')
                    print(f"Salvato checkpoint epoch {epoch}")

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_lr=lr
                    best_bs=bs
                    torch.save(model.state_dict(), 'checkpoints/best_model_lr{lr}_bs{bs}.pth')
                    print(f"Nuovo best model con acc: {best_acc:.2f}%")


            print(f" Fine training per lr={best_lr}, bs={best_bs} | Best Acc: {best_acc:.2f}%")
