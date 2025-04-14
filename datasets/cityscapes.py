import os
from glob import glob
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToPILImage, Resize, ToTensor, InterpolationMode, PILToTensor


class CityScapes(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        """
        Cityscapes dataset (immagine + maschera concatenate in un unico .jpg)

        Args:
            data_dir (str): directory base contenente train/, val/, test/
            split (str): uno tra 'train', 'val', 'test'
            transform (dict): dizionario con 'image' e 'mask' transformations
        """
        super(CityScapes, self).__init__()
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        self.image_paths = glob(os.path.join(self.data_dir, split, '*.jpg'))
        self.to_pil = ToPILImage()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Carica l'immagine concatenata (image | mask)
        full_image = cv2.imread(img_path)
        full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)

        # Divide in immagine e maschera
        w = full_image.shape[1] // 2
        image = full_image[:, :w, :]
        mask = full_image[:, w:, :]

        # Converti in PIL
        image = self.to_pil(image)
        mask = Image.fromarray(mask[:, :, 0])  # prende solo un canale come maschera

        # Applica trasformazioni
        if self.transform:
            image = self.transform['image'](image)
            mask = self.transform['mask'](mask)

        return image, mask, img_path

from torchvision import transforms

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

dataset = CityScapes(data_dir="/path/to/cityscapes_data", split="train", transform=transform)
