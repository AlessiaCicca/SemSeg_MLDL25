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


#----------------------------
#      CLASSES DEFINITION
#---------------------------

class GTA5(Dataset):
    def __init__(self, annotations_file, root_dir, transform=None, target_transform=None, mask_preprocessed_dir=None):
        self.img_labels = pd.read_csv(annotations_file)     #Reads the CSV file containing relative paths for images and masks
        self.root_dir = root_dir                             # Base directory where data is located
        self.transform = transform                           #Transformations to apply on the images
        self.target_transform = target_transform              #Transformations to apply on the masks
        self.mask_preprocessed_dir = mask_preprocessed_dir    # Path to the processed mask in train_id format

    def __len__(self):
        # Returns the number of samples in the dataset
        return len(self.img_labels)

    # This method loads the image and its corresponding mask at the given index idx. It constructs the full file paths from the CSV info,
    #opens both files (converting the image to RGB), applies any given transformations (like resizing or normalization), and returns them as a
    #tuple (image, mask) ready for training or evaluation

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

#----------------------------
#   DOWNLOAD OF THE DATASET
#---------------------------

#This code verifies whether the required dataset folder and its essential subfolders (images and labels) exist. 
#If any of them is missing, it notifies that the dataset is not available. The download is performed in the training


extract_dir = "/content/SemSeg_MLDL25/tmp/GTA5/GTA5"

def find_folder(start_path, folder_name):
    for root, dirs, _ in os.walk(start_path):
        if folder_name in dirs:
            return os.path.join(root, folder_name)
    return None

if not os.path.exists(extract_dir) or not find_folder(extract_dir, 'images') or not find_folder(extract_dir, 'labels'):
    print("Dataset not present")


#----------------------------
#   BASIC TRASFORMATION 
#---------------------------

#Images need normalization and resizing to standardize input for the model.
#Masks must be resized carefully without interpolation that changes label IDs (such as nearest neighbor interpolation ),
#and converted to tensor format the model expects.

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


#----------------------------
#   SPLIT OF THE DATASET 
#---------------------------

#This function filters and collects valid image-mask pairs from a list, preparing them with relative paths for further use 
#(like creating CSVs or feeding a dataset).

def split_train_val_gta5(image_files, masks_dir, root_dir, data):
    #Iterate over all the images  in image_files                                                                                                                   
    for img_mask_path in image_files:
        if os.path.exists(img_mask_path[0]) and os.path.exists(img_mask_path[1]):
            img_rel = os.path.relpath(img_mask_path[0], root_dir)
            mask_rel_val = os.path.relpath(img_mask_path[1], root_dir)
            data.append([img_rel, mask_rel_val])
        else:
            print(f"[WARNING] There isn't a mask for {img_mask_path[0]} or {img_mask_path[1]}")


#This function creates two CSV files (one for training and one for validation) containing pairs of image paths and their 
#corresponding mask paths from the GTA5 dataset. It splits the dataset randomly but reproducibly (using a fixed seed).

def create_gta5_csv(images_dir, masks_dir, output_train_csv, output_val_csv, root_dir):
'''
Inputs:

images_dir: folder containing all the images.
masks_dir: folder containing all the masks.
output_train_csv: filename/path to save the training CSV.
output_val_csv: filename/path to save the validation CSV.
root_dir: root directory used to compute relative paths.

'''
    print('gta5_dir : ', images_dir)
    #Recursively searches for all PNG files in the images and masks folders and stores them in two lists.
    image_files = glob(os.path.join(images_dir, '**', '*.png'), recursive=True)
    masks_files = glob(os.path.join(masks_dir, "**", "*.png"), recursive=True)

    #Converts the lists of file paths into Pandas DataFrames and concatenate them: column 0 = image path, column 1 = mask path
    image_files = pd.DataFrame(image_files)
    masks_files = pd.DataFrame(masks_files)
    images_masks = pd.concat([image_files, masks_files], axis=1)

     #Splits the combined DataFrame into training and validation sets, with 70% training and 30% validation. 
     #Shuffling is enabled with a fixed random seed for reproducibility.
    files_train, files_val = train_test_split(images_masks, test_size=0.3, shuffle = True, random_state = 42)
    files_train = files_train.values.tolist()
    files_val = files_val.values.tolist()
    
    data_train = []
    data_val = []
    print(f"{len(image_files)} images found.")

    #Train -> pass the images to create the csv file
    split_train_val_gta5(files_train, masks_dir, root_dir, data_train)

    #Validation -> pass the images to create the csv file
    split_train_val_gta5(files_val, masks_dir, root_dir, data_val)

    if len(data_train) == 0 or len(data_val) == 0:
        print("[ERROR] No pairs found!")
    
    #Write the data to a CSV file:
    #TRAIN
    with open(output_train_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'mask'])
        writer.writerows(data_train)
    #VALIDATION
    with open(output_val_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'mask'])
        writer.writerows(data_val)

    print(f"Created CSV file with {len(data_train)} pairs: {output_train_csv}")
    print(f"Created CSV file with {len(data_val)} pairs: {output_val_csv}")

