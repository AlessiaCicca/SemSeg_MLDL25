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
from augDoppio import CombinedAugmentation, val_transform_fn  # se lo metti in un file separato


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


if __name__ == "__main__":
    print(">>> Avvio training...")

    base_extract_path = './tmp/GTA5'
    zip_path = 'gt5_dataset.zip'

    # DA COMMENTARE SE SCARICATE IL FILE IN LOCALE
    gdrive_id = "1xYxlcMR2WFCpayNrW2-Rb7N-950vvl23"
    gdown_url = f"https://drive.google.com/uc?id={gdrive_id}"

    if not os.path.exists(base_extract_path):
        print("📦 Dataset non trovato o incompleto, lo scarico...")

        # Scarica il file dal link corretto
        gdown.download(gdown_url, zip_path, quiet=False)

        # Verifica che sia uno zip valido
        if zipfile.is_zipfile(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(base_extract_path)
            print("✅ Estrazione completata.")
        else:
            print("❌ Il file scaricato non è un file zip valido.")
            os.remove(zip_path)  # Elimina file corrotto
    else:
        print("✅ Dataset già presente.")

    train_images_dir = find_folder(base_extract_path, 'images')
    train_masks_dir = find_folder(base_extract_path, 'labels')

    train_csv = 'train_gta5_annotations.csv'
    val_csv = 'val_gta5_annotations.csv'

    GTA5.create_gta5_csv(train_images_dir, train_masks_dir, train_csv, val_csv, base_extract_path)

    preprocessed_masks_dir = './tmp/GTA5/GTA5/labels_trainid'  # cartella con maschere preprocessate

    base_train_dataset = GTA5.GTA5(
        annotations_file=train_csv,
        root_dir=base_extract_path,
        transform=None,
        target_transform=None,
        mask_preprocessed_dir=preprocessed_masks_dir
    )

    train_transform = CombinedAugmentation(dataset=base_train_dataset, crop_size=(512, 1024))


    train_dataset = GTA5.GTA5(
        annotations_file=train_csv,
        root_dir=base_extract_path,
        transform=train_transform,
        target_transform=None,
        mask_preprocessed_dir=preprocessed_masks_dir
    )

    val_dataset = GTA5.GTA5(
        annotations_file=val_csv,
        root_dir=base_extract_path,
        transform=val_transform_fn,
        target_transform=None,
        mask_preprocessed_dir=preprocessed_masks_dir
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs('checkpoints', exist_ok=True)

    num_epochs = 10
    lr = 0.000025
    bs = 4

    print(f"\n>>> Inizio training con lr={lr}, batch_size={bs}")
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True,
                              num_workers=2, pin_memory=True)

    start = time.time()
    for i, (inputs, targets) in enumerate(train_loader):
        if i == 5:  # solo 5 batch per test
            break
    end = time.time()
    print(f"Tempo per caricare 5 batch: {end - start:.2f} secondi")

    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False,
                            num_workers=2, pin_memory=True)

    model = BiSeNet(num_classes=19, context_path='resnet18').to(device)
    trainid_mask_dir = "./tmp/GTA5/GTA5/labels_trainid"
    class_weights = compute_class_weights(trainid_mask_dir).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    best_acc = 0
    best_miou = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train(epoch, model, train_loader, criterion, optimizer, device)
        val_acc, val_miou = validate(model, val_loader, criterion, device)

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            torch.save(model.state_dict(), f'checkpoints/checkpoint_epoch_{epoch}_lr{lr}_bs{bs}.pth')
            print(f"checkpoint epoch {epoch}")

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            print(f"Nuovo best model con mIoU: {best_miou:.2f}% (Acc: {val_acc:.2f}%)")

    torch.save(model.state_dict(), f'checkpoints/final_model_lr{lr}_bs{bs}.pth')
    print(f"Fine training per lr={lr}, bs={bs} | Best mIoU: {best_miou:.2f}%")
