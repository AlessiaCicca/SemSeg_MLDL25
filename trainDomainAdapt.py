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
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import subprocess
from focalLoss import FocalLossMulticlass
from models.bisenet.build_bisenet import BiSeNet
import datasets.gta5 as GTA5
from augmentation import CombinedAugmentation, val_transform_fn, val_transform_fn_no_mask
import datasets.cityscapes as cityscapes
from discriminator import FCDiscriminator

'''
Dictionary of available loss function options.
 Each entry maps a loss name to a lambda that initializes the corresponding loss function 
 with class weights and ignore index as parameters.
     - "CrossEntropy": Standard cross-entropy loss with class balancing.
     - "FocalLoss": Focal loss for addressing class imbalance, with gamma=2.0.
'''
criterion_options = {
    "CrossEntropy": lambda class_weights, ignore_index: nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index),
     "FocalLoss": lambda class_weights, ignore_index: FocalLossMulticlass(gamma=2.0, weight=class_weights, ignore_index=255)
}

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
    for mask_path in tqdm(mask_paths, desc="Class frequency computation"):
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
        ious.append(intersection / union)
    return np.mean(ious) if ious else 0.0

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

# ----------------------
# Training e Validation
# ----------------------
def train_adapt(epoch, model, discriminator, train_loader, target_loader,
                criterion, criterion_adv, optimizer_G, optimizer_D, device, lambda_adv):
    """
    Performs a single training epoch with domain adaptation.

    Args:
        epoch : Current epoch index (starting from 0).
        model : The segmentation network to be trained.
        discriminator : The discriminator network used for domain adaptation.
        train_loader : DataLoader providing source domain (labeled) data.
        target_loader : DataLoader providing target domain (unlabeled) data.
        criterion : Loss function for supervised segmentation.
        criterion_adv : Adversarial loss function for domain adaptation.
        optimizer_G : Optimizer for the segmentation network.
        optimizer_D : Optimizer for the discriminator network.
        device : Device on which computations are performed (CPU or GPU).
        lambda_adv : Weighting factor for the adversarial loss term.

    This function executes one training loop that combines supervised learning on the source domain
    with adversarial domain adaptation. The discriminator guides the segmentation network to learn 
    domain-invariant features by distinguishing source from target outputs, while the segmenter aims 
    to fool the discriminator on target data.
    
    """

    model.train()
    discriminator.train()
    running_seg, running_adv, running_D = 0.0, 0.0, 0.0
    correct, total = 0, 0
    target_iter = iter(target_loader)
    scaler = GradScaler()

    for inputs_src, targets_src in train_loader:
        inputs_src, targets_src = inputs_src.to(device), targets_src.to(device)
        try:
            inputs_tgt = next(target_iter)[0].to(device)
        except StopIteration:
            target_iter = iter(target_loader)
            inputs_tgt = next(target_iter)[0].to(device)

        with autocast('cuda'):
            out_src = model(inputs_src)[0]
            out_tgt = model(inputs_tgt)[0]
            loss_seg = criterion(out_src, targets_src.squeeze(1).long())
            soft_src = torch.softmax(out_src.detach(), dim=1)
            soft_tgt = torch.softmax(out_tgt.detach(), dim=1)
            pred_src = discriminator(soft_src)
            pred_tgt = discriminator(soft_tgt)
            loss_D = 0.5 * (
                criterion_adv(pred_src, torch.ones_like(pred_src)) +
                criterion_adv(pred_tgt, torch.zeros_like(pred_tgt))
            )

        optimizer_D.zero_grad()
        scaler.scale(loss_D).backward()
        scaler.step(optimizer_D)

        with autocast('cuda'):
            pred_tgt_for_G = discriminator(torch.softmax(out_tgt, dim=1))
            loss_adv = criterion_adv(pred_tgt_for_G, torch.ones_like(pred_tgt_for_G))
            loss_total = loss_seg + lambda_adv * loss_adv

        optimizer_G.zero_grad()
        scaler.scale(loss_total).backward()
        scaler.step(optimizer_G)
        scaler.update()

        running_seg += loss_seg.item()
        running_adv += loss_adv.item()
        running_D += loss_D.item()

        _, predicted = torch.max(out_src, 1)
        correct += (predicted == targets_src.squeeze(1)).sum().item()
        total += torch.numel(targets_src.squeeze(1))
        torch.cuda.empty_cache()

    acc = 100. * correct / total
    print(f"Train Epoch {epoch+1} - SegLoss: {running_seg/len(train_loader):.4f} - AdvLoss: {running_adv/len(train_loader):.4f} - DLoss: {running_D/len(train_loader):.4f} - Acc: {acc:.2f}%")

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
            miou_total += compute_miou(predicted, targets.squeeze(1), num_classes)
            count += 1
            gc.collect()

    acc = 100. * correct / total
    mean_iou = 100. * (miou_total / count) if count > 0 else 0
    print(f'Validation on Gta5- Loss: {val_loss / len(val_loader):.4f} - Acc: {acc:.2f}% - mIoU: {mean_iou:.2f}%')
    return acc, mean_iou

# ----------------------
# MAIN
# ----------------------
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)

    print(">>> Training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs('checkpoints_ce', exist_ok=True)

    # === Dataset GTA5
    base_extract_path = './tmp/GTA5'
    gdrive_id = "1xYxlcMR2WFCpayNrW2-Rb7N-950vvl23"
    zip_path = 'gt5_dataset.zip'

    if not os.path.exists(base_extract_path):
        gdown.download(f"https://drive.google.com/uc?id={gdrive_id}", zip_path, quiet=False)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(base_extract_path)

    train_images_dir = find_folder(base_extract_path, 'images')
    train_masks_dir = find_folder(base_extract_path, 'labels')
    train_csv = 'train_gta5_annotations.csv'
    val_csv = 'val_gta5_annotations.csv'
    GTA5.create_gta5_csv(train_images_dir, train_masks_dir, train_csv, val_csv, base_extract_path)
    result = subprocess.run(['python3', 'preprocess_mask.py'], capture_output=True, text=True)
    print("Output preprocess_mask.py:\n", result.stdout)
    if result.returncode != 0:
        print("Preprocess_mask.py failed:\n", result.stderr)
    else:
        print("Preprocess_mask.py ran successfully.")


    preprocessed_masks_dir = os.path.join(base_extract_path, 'GTA5', 'labels_trainid')

    base_train_dataset = GTA5.GTA5(train_csv, base_extract_path, None, None, preprocessed_masks_dir)
    train_transform = CombinedAugmentation(dataset=base_train_dataset, crop_size=(512, 1024))
    train_dataset = GTA5.GTA5(train_csv, base_extract_path, train_transform, None, preprocessed_masks_dir)
    val_dataset = GTA5.GTA5(val_csv, base_extract_path, val_transform_fn, None, preprocessed_masks_dir)

    # === Cityscapes (target domain)
    target_csv = 'cityscapes_target.csv'
    target_root = './Cityscapes/Cityscapes/Cityspaces/images/train' 
    cityscapes.create_csv_no_labels(target_root, target_csv)
    target_dataset = cityscapes.CityscapesNoLabel(
        annotations_file=target_csv,
        transform=val_transform_fn_no_mask
    )



    # === DataLoader
    bs = 4
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    target_loader = DataLoader(target_dataset, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    # === Training setup
    num_epochs = 50
    initial_lr = 2.5e-5
    class_weights = compute_class_weights(preprocessed_masks_dir).to(device)
    lambda_adv = 0.001

    for loss_name, criterion_fn in criterion_options.items():
        print(f"\nTraining with loss={loss_name}")

        model = BiSeNet(num_classes=19, context_path='resnet18').to(device)
        '''
        LOADING OF PRETRAINED WEIGHT
        pretrained_path = './checkpoints_pretrain/pretrain_irfs_bisenet_final.pth'
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Checkpoint preaddestrato non trovato in: {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=device)) 
        '''
        discriminator = FCDiscriminator(num_classes=19).to(device)

        optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
        optimizer_disc = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.9, 0.99))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / num_epochs) ** 0.9)
        scheduler_disc = optim.lr_scheduler.LambdaLR(optimizer_disc, lr_lambda=lambda epoch: (1 - epoch / num_epochs) ** 0.9)

        criterion = criterion_fn(class_weights, ignore_index=255).to(device)

       
        criterion_ad = nn.BCEWithLogitsLoss()

        best_miou = 0
        exp_name = f"loss_{loss_name}"
        exp_ckpt_dir = os.path.join("checkpoints_ce", exp_name)
        os.makedirs(exp_ckpt_dir, exist_ok=True)

        for epoch in range(num_epochs):
            print(f"\n[Exp: {exp_name}] Epoch {epoch+1}/{num_epochs}")
            train_adapt(epoch, model, discriminator, train_loader, target_loader,
                        criterion, criterion_ad, optimizer, optimizer_disc, device, lambda_adv)
            val_acc, val_miou = validate(model, val_loader, criterion, device)

            if val_miou > best_miou:
                best_miou = val_miou
                torch.save(model.state_dict(), os.path.join(exp_ckpt_dir, 'best_model_ce.pth'))
                print(f" New best model with mIoU: {best_miou:.2f}% (Acc: {val_acc:.2f}%)")

            scheduler.step()
            scheduler_disc.step()

        torch.save(model.state_dict(), os.path.join(exp_ckpt_dir, 'final_model_ce.pth'))
        print(f" Training completed: {exp_name} | Best mIoU: {best_miou:.2f}%")
