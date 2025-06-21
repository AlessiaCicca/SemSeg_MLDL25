import os
from PIL import Image
import numpy as np
from glob import glob
from tqdm import tqdm 

# Map RGB -> trainId
#This dictionary maps each RGB color (used as label colors in the original dataset) to an integer trainId.


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

# Create a 3D lookup table (256 x 256 x 256) initialized to 255 (ignore label).
# This table allows fast mapping from any RGB triplet to the corresponding trainId.
# The shape corresponds to all possible values of RGB colors (0-255 each).
lookup = np.ones((256, 256, 256), dtype=np.uint8) * 255  # default ignore 255
for rgb, train_id in rgb_to_trainid.items():
#we substitute the 255 with the train_id when we identify one of the RGB triplet of our dataset
    lookup[rgb] = train_id

def convert_mask_rgb_to_trainid(mask_path, save_path):
    #Load the image and convert to RGB format.
    mask = Image.open(mask_path).convert('RGB')
    #Convert the image to a numpy array for easy pixel-wise access
    mask_np = np.array(mask)
    # Use lookup table for fast RGB -> trainId conversion
    trainid_mask = lookup[mask_np[:,:,0], mask_np[:,:,1], mask_np[:,:,2]]
    # Convert the trainId array back to an image.
    trainid_img = Image.fromarray(trainid_mask)
    trainid_img.save(save_path)

if __name__ == "__main__":
    masks_dir = "./tmp/GTA5/GTA5/labels"  #original folder
    output_dir = "./tmp/GTA5/GTA5/labels_trainid"  #new folder

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    #Get list of all PNG files in the original folder
    mask_files = glob(os.path.join(masks_dir, "*.png"))

    #Convert each file of the list
    for mask_file in tqdm(mask_files):
        filename = os.path.basename(mask_file)
        save_path = os.path.join(output_dir, filename)
        convert_mask_rgb_to_trainid(mask_file, save_path)

    print("Preprocessing completed!")


