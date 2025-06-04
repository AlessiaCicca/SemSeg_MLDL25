import os
import zipfile
import gdown
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize, InterpolationMode, PILToTensor
import pandas as pd
import csv

#FATTO TUTTO DA NOI CAPIRE

# ------------------------------
# Dataset Class
# ------------------------------
class CityScapes(Dataset):
    def __init__(self, annotations_file, root_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_labels.iloc[idx, 0])
        mask_path = os.path.join(self.root_dir, self.img_labels.iloc[idx, 1])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask
    

class CityscapesNoLabel(Dataset):
    def __init__(self, annotations_file, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0] # path assoluto
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return (image,) 

# ------------------------------
# Scarica e prepara il dataset
# ------------------------------
# ------------------------------
# Parametri dataset
# ------------------------------
#download_url = 'https://drive.google.com/uc?id=1Qb4UrNsjvlU-wEsR9d7rckB0YS_LXgb2'
#output_zip = 'cityscapes_dataset.zip'
extract_dir = './tmp/Cityscapes'

# ------------------------------
# Funzione per cercare cartelle ricorsivamente
# ------------------------------
def find_folder(start_path, folder_name):
    for root, dirs, _ in os.walk(start_path):
        if folder_name in dirs:
            return os.path.join(root, folder_name)
    return None
'''
# ------------------------------
# Scarica ed estrai il dataset
# ------------------------------
if not os.path.exists(extract_dir) or not find_folder(extract_dir, 'images') or not find_folder(extract_dir, 'gtFine'):
    print("Dataset non trovato o incompleto, inizio il download...")
    if os.path.exists(extract_dir):
        print("Rimuovo versione incompleta...")
        os.system(f"rm -rf {extract_dir}")

    gdown.download(download_url, output_zip, quiet=False)

    print("Estrazione dell'archivio...")
    with zipfile.ZipFile(output_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    print("Download ed estrazione completati.")
else:
    print("✅ Dataset già presente.")

# ------------------------------
# Verifica cartelle estratte
# ------------------------------
extract_dir = './Cityscapes'
nested_path = os.path.join(extract_dir, 'Cityscapes', 'Cityscapes')
if os.path.exists(nested_path):
    extract_dir = nested_path # entra nel livello giusto

'''

# Rileva di nuovo le cartelle
images_dir = find_folder(extract_dir, 'images')
gtfine_dir = find_folder(extract_dir, 'gtFine')

if not images_dir or not gtfine_dir:
    raise RuntimeError("❌ Estrazione fallita: 'images' o 'gtFine' non trovati.")
else:
    print("✅ Dataset estratto correttamente.")
    print("Percorso immagini:", images_dir)
    print("Percorso maschere:", gtfine_dir)
# ------------------------------
# Trasformazioni
# ------------------------------

# VALUTARE SE EFFETTUARE ALTRE TRASFORMAZIONI
transform = {
    'image': transforms.Compose([
        Resize((512, 1024)),
        ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    'mask': transforms.Compose([
        Resize((512, 1024), interpolation=InterpolationMode.NEAREST),
        PILToTensor()
    ])
}

def create_cityscapes_csv(images_dir, masks_dir, output_csv, root_dir):
    image_files = glob(os.path.join(images_dir, '*', '*_leftImg8bit.png'), recursive=True)
    data = []

    print(f"Trovate {len(image_files)} immagini.")

    for img_path in image_files:
        basename = os.path.basename(img_path).replace('_leftImg8bit.png', '')
        city = os.path.basename(os.path.dirname(img_path))

        mask_filename = f"{basename}_gtFine_labelTrainIds.png"
        mask_path = os.path.join(masks_dir, city, mask_filename)

        if os.path.exists(mask_path):
            # Percorsi relativi rispetto alla root del dataset
            img_rel = os.path.relpath(img_path, root_dir)
            mask_rel = os.path.relpath(mask_path, root_dir)
            data.append([img_rel, mask_rel])
        else:
            print(f"[WARNING] Maschera mancante per {img_path}")

    if len(data) == 0:
        print("[ERROR] Nessuna coppia trovata!")

    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'mask'])
        writer.writerows(data)

    print(f"Creato CSV con {len(data)} coppie: {output_csv}")



def create_csv_no_labels(images_dir, output_csv):
    # Cerca ricorsivamente le immagini *_leftImg8bit.png
    image_files = glob(os.path.join(images_dir, '**', '*_leftImg8bit.png'), recursive=True)

    print(f"Trovate {len(image_files)} immagini.")

    if len(image_files) == 0:
        print("[ERROR] Nessuna immagine trovata!")

    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image']) # intestazione
        for img_path in image_files:
            writer.writerow([img_path]) # path assoluto

    print(f"Creato CSV con {len(image_files)} immagini: {output_csv}")
