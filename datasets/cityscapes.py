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
        img_path = self.img_labels.iloc[idx, 0]  # path assoluto
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return (image,)
        
def squeeze_and_long(x):
    return x.squeeze(0).long()
    
# Download dataset

download_url = 'https://drive.google.com/uc?id=1Qb4UrNsjvlU-wEsR9d7rckB0YS_LXgb2'
output_zip = 'cityscapes_dataset.zip'
extract_dir = './Cityscapes'


def find_folder(start_path, folder_name):
    for root, dirs, _ in os.walk(start_path):
        if folder_name in dirs:
            return os.path.join(root, folder_name)
    return None
    
if not os.path.exists(extract_dir) or not find_folder(extract_dir, 'images') or not find_folder(extract_dir, 'gtFine'):
    print("Dataset not present!")
    if os.path.exists(extract_dir):
        os.system(f"rm -rf {extract_dir}")

    gdown.download(download_url, output_zip, quiet=False)

    print("Extracting the dataset....")
    with zipfile.ZipFile(output_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    print("Download completed")



extract_dir = './Cityscapes'
nested_path = os.path.join(extract_dir, 'Cityscapes', 'Cityscapes')
if os.path.exists(nested_path):
    extract_dir = nested_path  # entra nel livello giusto
images_dir = find_folder(extract_dir, 'images')
gtfine_dir = find_folder(extract_dir, 'gtFine')

if not images_dir or not gtfine_dir:
    raise RuntimeError("'images' o 'gtFine' not fount.")

#Basic trasformation

transform = {
    'image': transforms.Compose([
        Resize((512, 1024)),
        ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    'mask': transforms.Compose([
        Resize((512, 1024), interpolation=InterpolationMode.NEAREST),
        PILToTensor(),
        #To 2b training
        #transforms.Lambda(squeeze_and_long) 
    ])
}

def create_cityscapes_csv(images_dir, masks_dir, output_csv, root_dir):
    image_files = glob(os.path.join(images_dir, '*', '*_leftImg8bit.png'), recursive=True)
    data = []

    print(f"There are {len(image_files)} images.")

    for img_path in image_files:
        basename = os.path.basename(img_path).replace('_leftImg8bit.png', '')
        city = os.path.basename(os.path.dirname(img_path))
        mask_filename = f"{basename}_gtFine_labelTrainIds.png"
        mask_path = os.path.join(masks_dir, city, mask_filename)
        if os.path.exists(mask_path):
            img_rel = os.path.relpath(img_path, root_dir)
            mask_rel = os.path.relpath(mask_path, root_dir)
            data.append([img_rel, mask_rel])
        else:
            print(f"[WARNING] there isn't the mask for {img_path}")

    if len(data) == 0:
        print("[ERROR] No pairs (images-mask) found!")

    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'mask'])
        writer.writerows(data)

    print(f"Created CSV file with {len(data)} pairs: {output_csv}")

def create_csv_no_labels(images_dir, output_csv):
    image_files = glob(os.path.join(images_dir, '**', '*_leftImg8bit.png'), recursive=True)
    print(f"There are {len(image_files)} images.")
    if len(image_files) == 0:
        print("[ERROR] No pairs (images-mask) found!")
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image'])
        for img_path in image_files:
            writer.writerow([img_path])
    print(f"Created CSV file with {len(image_files)} pairs: {output_csv}")

