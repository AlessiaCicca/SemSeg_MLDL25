import random
import torch
from PIL import Image
import numpy as np
from torchvision.transforms.functional import to_tensor, normalize
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

class CustomAugmentation:
    def __init__(self, dataset, crop_size=(512, 1024), rare_classes=[12, 13, 17, 18]):
        self.dataset = dataset  # il dataset completo per campionare la seconda immagine
        self.crop_size = crop_size
        self.rare_classes = rare_classes
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, image, mask):
        while True:
            idx = random.randint(0, len(self.dataset) - 1)
            image_b, mask_b = self.dataset[idx]

            # Se mask_b è PIL Image, convertilo in numpy
            if isinstance(mask_b, Image.Image):
                mask_b_np = np.array(mask_b)
            elif isinstance(mask_b, torch.Tensor):
                mask_b_np = mask_b.cpu().numpy()
            elif isinstance(mask_b, np.ndarray):
                mask_b_np = mask_b
            else:
                raise TypeError(f"mask_b deve essere un numpy array, torch tensor o PIL Image, ma è {type(mask_b)}")

            if any(np.any(mask_b_np == cls) for cls in self.rare_classes):
                break

        # Converti a PIL Image per il resize se non lo è già
        if not isinstance(mask_b, Image.Image):
            mask_b = Image.fromarray(mask_b_np.astype(np.uint8))
        if isinstance(image_b, torch.Tensor):
            image_b = Image.fromarray((image_b.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

        target_size = (self.crop_size[0], self.crop_size[1])
        image = resize(image, target_size)
        image_b = resize(image_b, target_size)
        mask = resize(mask, target_size, interpolation=InterpolationMode.NEAREST)
        mask_b = resize(mask_b, target_size, interpolation=InterpolationMode.NEAREST)

        mask_b_np = np.array(mask_b)
        class_mask = np.zeros_like(mask_b_np, dtype=np.uint8)
        for cls in self.rare_classes:
            class_mask[mask_b_np == cls] = 1
        class_mask = torch.from_numpy(class_mask).float().unsqueeze(0)

        image = to_tensor(image)
        image_b = to_tensor(image_b)
        image = normalize(image, self.mean, self.std)
        image_b = normalize(image_b, self.mean, self.std)

        mixed_image = image * (1 - class_mask) + image_b * class_mask

        mask = torch.from_numpy(np.array(mask)).long()
        mask_b = torch.from_numpy(mask_b_np).long()
        mixed_mask = mask * (1 - class_mask.squeeze(0).long()) + mask_b * class_mask.squeeze(0).long()

        return mixed_image, mixed_mask


def val_transform_fn(image, mask):
    target_size = (512, 1024)
    image = resize(image, target_size)
    image = to_tensor(image)
    image = normalize(image, mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    mask = resize(mask, target_size, interpolation=InterpolationMode.NEAREST)
    mask = to_tensor(mask).long()
    mask = mask.squeeze(0)  # rimuovi canale extra se presente, mask shape: [H, W]

    return image, mask
