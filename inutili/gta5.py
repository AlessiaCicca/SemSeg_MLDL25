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

#Map RGB → trainId 
rgb_to_trainid = {
    (128, 64,128): 0,   # road
    (244, 35,232): 1,   # sidewalk
    (70, 70, 70): 2,    # building
    (102,102,156): 3,   # wall
    (190,153,153): 4,   # fence
    (153,153,153): 5,   # pole
    (250,170, 30): 6,   # traffic light
    (220,220,  0): 7,   # traffic sign
    (107,142, 35): 8,   # vegetation
    (152,251,152): 9,   # terrain
    (70,130,180): 10,   # sky
    (220, 20, 60): 11,  # person
    (255,  0,  0): 12,  # rider
    (0,   0, 142): 13,  # car
    (0,   0,  70): 14,  # truck
    (0,  60, 100): 15,  # bus
    (0,  80, 100): 16,  # train
    (0,   0, 230): 17,  # motorcycle
    (119, 11, 32): 18,  # bicycle
}

class GTA5(Dataset):
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
        
        mask = Image.open(mask_path).convert("RGB")
        mask = np.array(mask)

        h, w, _ = mask.shape
        new_mask = np.ones((h, w), dtype=np.uint8) * 255  # default = ignore
        for rgb, train_id in rgb_to_trainid.items():
            matches = np.all(mask == rgb, axis=-1)
            new_mask[matches] = train_id

    
        mask = Image.fromarray(new_mask)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        # Assicura tipo long per compatibilità con CrossEntropyLoss
        mask = mask.long()

        return image, mask

#Download dataset
extract_dir = r'.\\tmp\\GTA5'

def find_folder(start_path, folder_name):
    for root, dirs, _ in os.walk(start_path):
        if folder_name in dirs:
            return os.path.join(root, folder_name)
    return None

if not os.path.exists(extract_dir) or not find_folder(extract_dir, 'images') or not find_folder(extract_dir, 'labels'):
    print("Dataset present")
else:
    print("Dataset not present")

#Trasforamtion

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


def split_train_val_gta5(image_files, masks_dir, root_dir, data):

    for img_mask_path in image_files:

        if os.path.exists(img_mask_path[0]) and os.path.exists(img_mask_path[1]):
            img_rel = os.path.relpath(img_mask_path[0], root_dir)
            mask_rel_val = os.path.relpath(img_mask_path[1], root_dir)
            data.append([img_rel, mask_rel_val])
        else:
            print(f"[WARNING] There isn't a mask for {img_mask_path[0]} or {img_mask_path[1]}")



def create_gta5_csv(images_dir, masks_dir, output_train_csv, output_val_csv, root_dir):

    print('gta5_dir : ', images_dir)
    image_files = glob(os.path.join(images_dir, '**', '*.png'), recursive=True)
    masks_files = glob(os.path.join(masks_dir, "**", "*.png"), recursive=True)

    image_files = pd.DataFrame(image_files)
    masks_files = pd.DataFrame(masks_files)

    images_masks = pd.concat([image_files, masks_files], axis=1)

    files_train, files_val = train_test_split(images_masks, test_size=0.3, shuffle = True, random_state = 42)

    files_train = files_train.values.tolist()
    files_val = files_val.values.tolist()
    
    data_train = []
    data_val = []

    print(f"Trovate {len(image_files)} immagini.")

    #Train
    split_train_val_gta5(files_train, masks_dir, root_dir, data_train)

    #Validation
    split_train_val_gta5(files_val, masks_dir, root_dir, data_val)

        

    if len(data_train) == 0 or len(data_val) == 0:
        print("[ERROR] No pairs found!")
    

    with open(output_train_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'mask'])
        writer.writerows(data_train)

    
    with open(output_val_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'mask'])
        writer.writerows(data_val)

    print(f"Created CSV file with {len(data_train)} pairs: {output_train_csv}")
    print(f"Created CSV file with {len(data_val)} pairs: {output_val_csv}")

