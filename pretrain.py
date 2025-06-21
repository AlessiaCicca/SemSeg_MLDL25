import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from models.bisenet.build_bisenet import BiSeNet
import datasets.gta5 as GTA5
from augmentation import CombinedAugmentation
from PIL import Image
from time import time
import gdown
import zipfile
import subprocess
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import json
from torch.utils.tensorboard import SummaryWriter

SAVE_SELECTED_IMAGES_DIR = './selected_rare_class_images'
INFO_OUTPUT_DIR = './rare_class_info'
TENSORBOARD_LOGDIR = './runs/pretrain_irfs'

CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle"
]

os.makedirs(SAVE_SELECTED_IMAGES_DIR, exist_ok=True)
os.makedirs(INFO_OUTPUT_DIR, exist_ok=True)
os.makedirs('./checkpoints_pretrain', exist_ok=True)
os.makedirs(TENSORBOARD_LOGDIR, exist_ok=True)

writer = SummaryWriter(log_dir=TENSORBOARD_LOGDIR)


def analyze_rare_classes(frequencies, threshold=0.2, output_dir=INFO_OUTPUT_DIR):
    rare_classes = [i for i, f in enumerate(frequencies) if f < threshold]
    rare_dict = {CITYSCAPES_CLASSES[i]: float(frequencies[i]) for i in range(len(frequencies))}

    with open(os.path.join(output_dir, "class_frequencies_named.json"), "w") as f:
        json.dump(rare_dict, f, indent=4)

    colors = ['red' if i in rare_classes else 'blue' for i in range(len(frequencies))]
    plt.figure(figsize=(12, 6))
    plt.bar(CITYSCAPES_CLASSES, frequencies, color=colors)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Frequenza (presenza nelle immagini)")
    plt.title("Frequenze delle Classi (rosso = rare)")
    plt.tight_layout()
    plt.grid(True)
    plot_path = os.path.join(output_dir, "frequenze_classi_nome.png")
    plt.savefig(plot_path)
    plt.close()

    return rare_classes, plot_path


def get_rare_classes(dataset, num_classes=19, threshold=0.2):
    class_image_count = np.zeros(num_classes)
    total_images = len(dataset)

    for i in tqdm(range(total_images), desc="Analisi frequenze classi"):
        _, mask = dataset[i]
        mask_np = np.array(mask)
        for cls in np.unique(mask_np):
            if cls < num_classes:
                class_image_count[cls] += 1

    freq = class_image_count / total_images
    rare_classes, plot_path = analyze_rare_classes(freq, threshold)

    print(f"\nClassi rare identificate: {[CITYSCAPES_CLASSES[c] for c in rare_classes]}")
    print(f" Grafico salvato in: {plot_path}")
    return rare_classes, freq


def compute_miou_per_class(preds, labels, num_classes=19, ignore_index=255):
    ious = np.full(num_classes, np.nan)
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_inds = (preds == cls)
        label_inds = (labels == cls)
        intersection = np.logical_and(pred_inds, label_inds).sum()
        union = np.logical_or(pred_inds, label_inds).sum()
        if union > 0:
            ious[cls] = intersection / union
    return ious


def compute_irfs_factors(dataset, rare_classes, num_classes=19, t=1e-3):
    class_image_count = np.zeros(num_classes)
    class_instance_count = np.zeros(num_classes)
    total_images = len(dataset)

    for i in tqdm(range(total_images), desc="Calcolo IRFS"):
        _, mask = dataset[i]
        mask_np = np.array(mask)
        for cls in np.unique(mask_np):
            if cls < num_classes:
                class_image_count[cls] += 1
                class_instance_count[cls] += (mask_np == cls).sum() > 0

    f_image = class_image_count / total_images
    f_instance = class_instance_count / total_images
    r_c = np.maximum(1.0, np.sqrt(t / np.sqrt(f_image * f_instance + 1e-6)))

    repeat_factors = []
    for i in range(total_images):
        _, mask = dataset[i]
        mask_np = np.array(mask)
        r_i = max([r_c[c] for c in np.unique(mask_np) if c < num_classes], default=1.0)
        repeat_factors.append(int(np.ceil(r_i)))

    return repeat_factors


class IRFSDataset(Dataset):
    def __init__(self, base_dataset, repeat_factors, rare_classes):
        self.base_dataset = base_dataset
        self.repeat_factors = repeat_factors
        self.rare_classes = rare_classes
        self.indices = self._expand_indices()

    def _expand_indices(self):
        expanded = []
        for idx, rf in enumerate(self.repeat_factors):
            _, mask = self.base_dataset[idx]
            mask_np = np.array(mask)
            if any(cls in mask_np for cls in self.rare_classes):
                expanded.extend([idx] * rf)
        return expanded

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]



def mask_non_rare_targets(targets, rare_classes, ignore_index=255):
    masked = targets.clone()
    keep = torch.zeros_like(masked, dtype=torch.bool)
    for cls in rare_classes:
        keep |= (masked == cls)
    masked[~keep] = ignore_index
    return masked

def compute_class_weights(label_dir, num_classes=19):
    class_pixel_counts = np.zeros(num_classes, dtype=np.int64)
    mask_paths = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.png')]

    for mask_path in tqdm(mask_paths, desc="Calcolo pesi classi"):
        mask = np.array(Image.open(mask_path))
        for class_id in range(num_classes):
            class_pixel_counts[class_id] += np.sum(mask == class_id)

    total_pixels = np.sum(class_pixel_counts)
    class_freqs = class_pixel_counts / total_pixels
    weights = 1.0 / (np.log(1.02 + class_freqs))
    return torch.FloatTensor(weights)


def pretrain_model():
    base_path = './tmp/GTA5'
    train_csv = './train_gta5_annotations.csv'
    preprocessed_masks_dir = os.path.join(base_path, 'GTA5/labels_trainid')

    base_dataset = GTA5.GTA5(train_csv, base_path, transform=None, target_transform=None,
                             mask_preprocessed_dir=preprocessed_masks_dir)

    transform = CombinedAugmentation(dataset=base_dataset, crop_size=(512, 1024))
    augmented_dataset = GTA5.GTA5(train_csv, base_path, transform=transform, target_transform=None,
                                  mask_preprocessed_dir=preprocessed_masks_dir)

    rare_classes, freq = get_rare_classes(base_dataset)
    repeat_factors = compute_irfs_factors(base_dataset, rare_classes)

    train_dataset = IRFSDataset(augmented_dataset, repeat_factors, rare_classes)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiSeNet(num_classes=19, context_path='resnet18').to(device)

    class_weights = compute_class_weights(preprocessed_masks_dir).to(device)
    unweighted_criterion = nn.CrossEntropyLoss(ignore_index=255)
    weighted_criterion = nn.CrossEntropyLoss(ignore_index=255, weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 15
    for epoch in range(num_epochs):
        start_time = time()
        model.train()
        epoch_loss, correct, total = 0, 0, 0
        miou_acc = np.zeros(19)
        valid_cls = np.zeros(19)

        # Usa la loss pesata solo nelle epoche in cui mascheriamo per le rare
        using_weighted_loss = (epoch < 6)
        criterion = unweighted_criterion if epoch < 6 else weighted_criterion

        print(f"\n Epoch {epoch+1}/{num_epochs} | {' SOLO RARE (mascherate, pesata)' if using_weighted_loss else ' Tutte le classi (loss normale)'}")

        for images, masks in tqdm(train_loader, desc=f"Ep {epoch+1:02d}"):
            images, masks = images.to(device), masks.to(device).squeeze(1)
            if using_weighted_loss:
                masks = mask_non_rare_targets(masks, rare_classes)

            outputs = model(images)[0]
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == masks).sum().item()
            total += masks.numel()

            miou = compute_miou_per_class(preds, masks)
            for i, v in enumerate(miou):
                if not np.isnan(v):
                    miou_acc[i] += v
                    valid_cls[i] += 1

        # Log fine epoca
        acc = 100. * correct / total
        avg_loss = epoch_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Accuracy/train", acc, epoch)

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | Time needed: {timedelta(seconds=int(time() - start_time))}")

        for i in range(19):
            if valid_cls[i] > 0:
                mean_iou_class = miou_acc[i] / valid_cls[i]
                writer.add_scalar(f"mIoU/Class_{CITYSCAPES_CLASSES[i]}", mean_iou_class, epoch)

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            torch.save(model.state_dict(), f"checkpoints_pretrain/pretrain_epoch{epoch}_bisenet.pth")

    torch.save(model.state_dict(), "checkpoints_pretrain/pretrain_irfs_bisenet_final.pth")
    print("Pretraining completed")

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
      pretrain_model()
      writer.close()
