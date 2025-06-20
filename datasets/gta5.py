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
    def __init__(self, annotations_file, root_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

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

    print(f"{len(image_files)} images found.")

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

