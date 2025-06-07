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
from binaryfocal import FocalLossMulticlass
import numpy as np
import subprocess
from models.bisenet.build_bisenet import BiSeNet
import datasets.gta5WithoutRGB as GTA5
from augDoppioDA import CombinedAugmentation, val_transform_fn_no_mask, val_transform_fn # se lo metti in un file separato
import cityscapesDA as cityscapes
from discriminator import FCDiscriminator
import torch.nn.functional as F

scaler = GradScaler()  

# Implementazione semplice di DiceLoss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        inputs: logits [B, C, H, W]
        targets: class indices [B, H, W]
        """
        num_classes = inputs.shape[1]

        # One-hot encoding dei target: [B, H, W] -> [B, C, H, W]
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        # Softmax su input logits
        inputs_soft = F.softmax(inputs, dim=1)

        # Calcolo Dice per ogni classe
        dims = (0, 2, 3)  # somma su batch e spatial
        intersection = torch.sum(inputs_soft * targets_one_hot, dims)
        union = torch.sum(inputs_soft + targets_one_hot, dims)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        # Media sulle classi
        return 1 - dice.mean()


#criterion_options = {
    #"CrossEntropy": lambda class_weights, ignore_index: nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index),
    #"DiceLoss": lambda class_weights, ignore_index: DiceLoss(),
 #   "FocalLoss": lambda class_weights, ignore_index: FocalLossMulticlass(gamma=2.0)
#}
criterion_options = {
    "FocalLoss": lambda class_weights, ignore_index:  FocalLossMulticlass(gamma=2.0, weight=class_weights, ignore_index=255)
}


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


def train_adapt(epoch, model, discriminator, train_loader, target_loader,
                criterion, criterion_adv, optimizer_G, optimizer_D,
                device, lambda_adv):
    
    model.train()
    discriminator.train()
    running_seg, running_adv, running_D = 0.0, 0.0, 0.0
    correct, total = 0, 0

    # Iteratore per il dominio target (immagini senza label)
    target_iter = iter(target_loader)

    for inputs_src, targets_src in train_loader:
        inputs_src, targets_src = inputs_src.to(device), targets_src.to(device)

        try:
            inputs_tgt = next(target_iter)[0].to(device)
        except StopIteration:
            target_iter = iter(target_loader)
            inputs_tgt = next(target_iter)[0].to(device)

        # ======== FORWARD & DISCRIMINATOR TRAINING ========
        with autocast('cuda'):
            out_src = model(inputs_src)[0] # output della segmentazione source
            out_tgt = model(inputs_tgt)[0] # output della segmentazione target

            # Loss supervisata sul dominio source
            loss_seg = criterion(out_src, targets_src.squeeze(1).long())

            # Discriminatore: softmax su output (detach per evitare backprop in G)
            soft_src = torch.softmax(out_src.detach(), dim=1)
            soft_tgt = torch.softmax(out_tgt.detach(), dim=1)

            # fornisco al discriminatore le predizioni fatte sia sul source che sul target e ottengo una stima (cerca di indovinare se provengono dal dominio source o target)
            pred_src = discriminator(soft_src)
            pred_tgt = discriminator(soft_tgt)

            # La loss del discriminatore punisce gli errori nel distinguere le predizioni dei due domini.
            loss_D = 0.5 * (
                criterion_adv(pred_src, torch.ones_like(pred_src)) + # "vero" se source
                criterion_adv(pred_tgt, torch.zeros_like(pred_tgt)) # "falso" se target
            )

        # === BACKWARD discriminatore === Qui aggiorni i pesi di D per migliorare la sua capacità di distinguere source e target.
        ''' 
        Perché prima back su discriminatore e poi sul segmentatore?

        Perché nel passo successivo, il segmentatore proverà a ingannare D.
        Se aggiornassi prima G, poi D, il discriminatore riceverebbe gradienti “inquinati” da G non ancora stabile. D deve imparare a distinguere da G nella sua forma attuale, prima che G tenti di ingannarlo. 
        
        '''

        optimizer_D.zero_grad()
        scaler.scale(loss_D).backward()
        scaler.step(optimizer_D)

        # === BACKWARD segmentatore (G) ===
        with autocast('cuda'):
            soft_tgt_for_G = torch.softmax(out_tgt, dim=1)
            pred_tgt_for_G = discriminator(soft_tgt_for_G)

            loss_adv = criterion_adv(pred_tgt_for_G, torch.ones_like(pred_tgt_for_G))
            loss_total = loss_seg + lambda_adv * loss_adv

        optimizer_G.zero_grad()
        scaler.scale(loss_total).backward()
        scaler.step(optimizer_G)
        scaler.update()

        # === Logging ===
        running_seg += loss_seg.item()
        running_adv += loss_adv.item()
        running_D += loss_D.item()

        # usiamo solo il dominio source per calcolare l'accuratezza
        _, predicted = torch.max(out_src, 1)
        correct += (predicted == targets_src.squeeze(1)).sum().item()
        total += torch.numel(targets_src.squeeze(1))

        del inputs_src, targets_src, inputs_tgt, out_src, out_tgt
        torch.cuda.empty_cache()

    acc = 100. * correct / total
    print(f"Train Epoch {epoch} - SegLoss: {running_seg/len(train_loader):.4f} - AdvLoss: {running_adv/len(train_loader):.4f} - DLoss: {running_D/len(train_loader):.4f} - Acc: {acc:.2f}%")



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
    '''
    # Esegue lo script preprocess_mask.py
    result = subprocess.run(['python3', 'preprocess_mask.py'], capture_output=True, text=True)
    print("Output preprocess_mask.py:\n", result.stdout)
    if result.stderr:
        print("Errori preprocess_mask.py:\n", result.stderr)
    '''
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

    target_csv = 'cityscapes_target.csv'
    target_root = './Cityscapes/Cityscapes/Cityspaces/images'
    cityscapes.create_csv_no_labels(target_root, target_csv)

    target_dataset = cityscapes.CityscapesNoLabel(
        annotations_file=target_csv,
        transform=val_transform_fn_no_mask
    )



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs('checkpoints', exist_ok=True)

num_epochs = 10
lr = 0.000025
bs = 4

print(f"\n>>> Inizio training multi-esperimento con lr={lr}, batch_size={bs}")
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)
target_loader = DataLoader(target_dataset, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)

model_tmp = BiSeNet(num_classes=19, context_path='resnet18').to(device)  # temporaneo per pesi
trainid_mask_dir = "./tmp/GTA5/GTA5/labels_trainid"
class_weights = compute_class_weights(trainid_mask_dir).to(device)
#criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
del model_tmp  # non serve più

#lambda_adv_list = [0.001, 0.002, 0.005]
lambda_adv_list = [0.001]
use_weighted_bce_options = [ True]

for use_weighted_bce in use_weighted_bce_options:
    for lambda_adv in lambda_adv_list:
        for loss_name, criterion_fn in criterion_options.items():
            print(f"\n====== Esperimento: lambda_adv={lambda_adv}, loss={loss_name}, BCE weighted={use_weighted_bce} ======")

            model = BiSeNet(num_classes=19, context_path='resnet18').to(device)
            #model.load_state_dict(torch.load('/content/SemSeg_MLDL25/checkpoints/lambda0.001_lossFocalLoss_BCEweightedTrue/best_model.pth', map_location=device)) #da cancellare

            discriminator = FCDiscriminator(num_classes=19).to(device)

            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.9, 0.99))

            criterion = criterion_fn(class_weights, ignore_index=255).to(device)

            if use_weighted_bce:
                pos_weight = torch.tensor([2.0]).to(device)
                criterion_ad = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                criterion_ad = nn.BCEWithLogitsLoss()

            best_miou = 0
            exp_name = f"lambda{lambda_adv}_loss{loss_name}_BCEweighted{use_weighted_bce}"
            exp_ckpt_dir = os.path.join("checkpoints", exp_name)
            os.makedirs(exp_ckpt_dir, exist_ok=True)

            for epoch in range(num_epochs):
                print(f"\n[Exp: {exp_name}] Epoch {epoch}/{num_epochs}")
                train_adapt(epoch, model, discriminator, train_loader, target_loader,
                            criterion, criterion_ad, optimizer, optimizer_disc, device, lambda_adv)

                val_acc, val_miou = validate(model, val_loader, criterion, device)

                if val_miou > best_miou:
                    best_miou = val_miou
                    torch.save(model.state_dict(), os.path.join(exp_ckpt_dir, 'best_model.pth'))
                    print(f"✔️ Nuovo best model con mIoU: {best_miou:.2f}% (Acc: {val_acc:.2f}%)")

                if epoch % 10 == 0 or epoch == num_epochs - 1:
                    torch.save(model.state_dict(), os.path.join(exp_ckpt_dir, f'checkpoint_epoch_{epoch}.pth'))

            torch.save(model.state_dict(), os.path.join(exp_ckpt_dir, 'final_model.pth'))
            print(f"🏁 Fine esperimento: {exp_name} | Best mIoU: {best_miou:.2f}%")
