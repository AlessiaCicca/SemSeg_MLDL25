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
import numpy as np
from sklearn.model_selection import train_test_split



class GTA5(Dataset):
    def __init__(self, annotations_file, root_dir, transform=None, target_transform=None, mask_preprocessed_dir=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.mask_preprocessed_dir = mask_preprocessed_dir  # cartella maschere preprocessate (trainId)

    def __len__(self):
        return len(self.img_labels)

    '''

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_labels.iloc[idx, 0])

        if self.mask_preprocessed_dir:
            # Carica maschera preprocessata (singolo canale, valori trainId)
            mask_filename = os.path.basename(self.img_labels.iloc[idx, 1])
            mask_path = os.path.join(self.mask_preprocessed_dir, mask_filename)
            mask = Image.open(mask_path)  # modalità 'L' (8-bit)
        else:
            # Maschera RGB (vecchio metodo)
            mask_path = os.path.join(self.root_dir, self.img_labels.iloc[idx, 1])
            mask = Image.open(mask_path).convert("RGB")

            # Qui eventualmente la conversione RGB -> trainId (ma ora la togliamo se usiamo preprocessate)
            # ... conversione da RGB a trainId se serve

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        # Assicura tipo long per compatibilità con CrossEntropyLoss
        mask = mask.long()

        return image, mask
    '''
    def __getitem__(self, idx):
      img_path = os.path.join(self.root_dir, self.img_labels.iloc[idx, 0])

      if self.mask_preprocessed_dir:
          mask_filename = os.path.basename(self.img_labels.iloc[idx, 1])
          mask_path = os.path.join(self.mask_preprocessed_dir, mask_filename)
          mask = Image.open(mask_path)
      else:
          mask_path = os.path.join(self.root_dir, self.img_labels.iloc[idx, 1])
          mask = Image.open(mask_path).convert("RGB")

      image = Image.open(img_path).convert("RGB")

      if self.transform:
          image, mask = self.transform(image, mask)

      return image, mask
# ------------------------------
# Scarica e prepara il dataset
# ------------------------------
# ------------------------------
# Parametri dataset
# ------------------------------

extract_dir = r'.\\tmp\\GTA5'

# ------------------------------
# Funzione per cercare cartelle ricorsivamente
# ------------------------------
def find_folder(start_path, folder_name):
    for root, dirs, _ in os.walk(start_path):
        if folder_name in dirs:
            return os.path.join(root, folder_name)
    return None

# ------------------------------
# Scarica ed estrai il dataset
# ------------------------------
if not os.path.exists(extract_dir) or not find_folder(extract_dir, 'images') or not find_folder(extract_dir, 'labels'):

    print("Dataset non presente")
else:
    print("✅ Dataset già presente.")

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

# ------------------------------
# Crea CSV con immagini e maschere - images_dir, masks_dir, output_csv, root_dir

'''
riadatto la funzione per creare i file csv partendo dal training set

- spitto il traning set
- creo i file csv dalle partizioni train + val


'''
def split_train_val_gta5(image_files, masks_dir, root_dir, data):

    for img_mask_path in image_files:

        '''print ('sono dentro il primo for')
        basename = os.path.basename(img_path)
        
    
        mask_path = os.path.join(masks_dir, basename)'''


        #print ('prima di entrare nel for')

        if os.path.exists(img_mask_path[0]) and os.path.exists(img_mask_path[1]):

           # print('sono dentro il for')
            # Percorsi relativi rispetto alla root del dataset
            img_rel = os.path.relpath(img_mask_path[0], root_dir)
            mask_rel_val = os.path.relpath(img_mask_path[1], root_dir)


            data.append([img_rel, mask_rel_val])

            #print('dati appesi')
        else:
            print(f"[WARNING] Maschera mancante per {img_mask_path[0]} o {img_mask_path[1]}")


# ------------------------------
def create_gta5_csv(images_dir, masks_dir, output_train_csv, output_val_csv, root_dir):

    print('gta5_dir : ', images_dir)
    image_files = glob(os.path.join(images_dir, '**', '*.png'), recursive=True)
    masks_files = glob(os.path.join(masks_dir, "**", "*.png"), recursive=True)

    # coverto in dataframe

    image_files = pd.DataFrame(image_files)
    masks_files = pd.DataFrame(masks_files)

    images_masks = pd.concat([image_files, masks_files], axis=1)

    files_train, files_val = train_test_split(images_masks, test_size=0.3, shuffle = True, random_state = 42)

    files_train = files_train.values.tolist()
    files_val = files_val.values.tolist()
    
    data_train = []
    data_val = []

    print(f"Trovate {len(image_files)} immagini.")

    # ciclo per il train
    split_train_val_gta5(files_train, masks_dir, root_dir, data_train)

    #ciclo per il val
    split_train_val_gta5(files_val, masks_dir, root_dir, data_val)

        

    if len(data_train) == 0 or len(data_val) == 0:
        print("[ERROR] Nessuna coppia trovata!")
    

    with open(output_train_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'mask'])
        writer.writerows(data_train)

    
    with open(output_val_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'mask'])
        writer.writerows(data_val)

    print(f"Creato CSV con {len(data_train)} coppie: {output_train_csv}")
    print(f"Creato CSV con {len(data_val)} coppie: {output_val_csv}")

