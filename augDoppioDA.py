import random
import numbers
import math
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.transforms import PILToTensor, InterpolationMode
from torchvision.transforms import InterpolationMode
import random
import torch
from PIL import Image
import numpy as np
from torchvision.transforms import InterpolationMode, PILToTensor, Normalize
from torchvision.transforms.functional import to_tensor
from torchvision.transforms.functional import normalize


# Qui assumo che le classi RandomSized, AdjustGamma, AdjustSaturation, AdjustHue, AdjustBrightness, AdjustContrast
# siano già definite come nel tuo codice precedente, oppure le definisco sotto.

class AdjustGamma(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, img, mask):
        assert img.size == mask.size
        return F.adjust_gamma(img, random.uniform(1, 1 + self.gamma)), mask

class AdjustSaturation(object):
    def __init__(self, saturation):
        self.saturation = saturation

    def __call__(self, img, mask):
        assert img.size == mask.size
        return (
            F.adjust_saturation(img, random.uniform(1 - self.saturation, 1 + self.saturation)),
            mask,
        )

class AdjustHue(object):
    def __init__(self, hue):
        self.hue = hue

    def __call__(self, img, mask):
        assert img.size == mask.size
        return F.adjust_hue(img, random.uniform(-self.hue, self.hue)), mask

class AdjustBrightness(object):
    def __init__(self, bf):
        self.bf = bf

    def __call__(self, img, mask):
        assert img.size == mask.size
        return F.adjust_brightness(img, random.uniform(1 - self.bf, 1 + self.bf)), mask

class AdjustContrast(object):
    def __init__(self, cf):
        self.cf = cf

    def __call__(self, img, mask):
        assert img.size == mask.size
        return F.adjust_contrast(img, random.uniform(1 - self.cf, 1 + self.cf)), mask

class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, mask
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return (img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST))
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return (img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST))

class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.0))
        y1 = int(round((h - th) / 2.0))
        return (img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th)))

class RandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return (img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST))

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return (img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th)))

class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, img, mask):
        assert img.size == mask.size

        w = int(random.uniform(0.5, 2) * img.size[0])
        h = int(random.uniform(0.5, 2) * img.size[1])

        img, mask = (img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST))

        return self.crop(*self.scale(img, mask))

class CombinedAugmentation:
    def __init__(self, dataset, crop_size=(512, 1024), rare_classes=[12, 13, 17, 18], scale_choices=[0.75, 1.0, 1.5, 1.75, 2.0]):
        self.dataset = dataset
        self.crop_size = crop_size
        self.rare_classes = rare_classes
        self.scale_choices = scale_choices

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.random_sized = RandomSized(crop_size[0])
        self.adjust_gamma = AdjustGamma(gamma=0.5)
        self.adjust_saturation = AdjustSaturation(saturation=0.5)
        self.adjust_hue = AdjustHue(hue=0.1)
        self.adjust_brightness = AdjustBrightness(bf=0.3)
        self.adjust_contrast = AdjustContrast(cf=0.3)

    def __call__(self, image: Image.Image, mask: Image.Image):
        # 1) Applicare prima le trasformazioni geometriche e colore alla immagine e mask input
        scale = random.choice(self.scale_choices)
        new_size = (int(image.height * scale), int(image.width * scale))
        image = F.resize(image, new_size)
        mask = F.resize(mask, new_size, interpolation=InterpolationMode.NEAREST)

        if random.random() > 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)

        angle = random.uniform(-10, 10)
        image = F.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
        mask = F.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

        image, mask = self.random_sized(image, mask)

        color_transforms = [
            self.adjust_gamma,
            self.adjust_saturation,
            self.adjust_hue,
            self.adjust_brightness,
            self.adjust_contrast,
        ]
        random.shuffle(color_transforms)
        for t in color_transforms:
            image, mask = t(image, mask)

        # 2) Ora applica il mixing con immagine e mask da dataset in base alle rare_classes

        while True:
            idx = random.randint(0, len(self.dataset) - 1)
            image_b, mask_b = self.dataset[idx]

            # Convert mask_b in numpy array per verifica rare classes
            if isinstance(mask_b, Image.Image):
                mask_b_np = np.array(mask_b)
            elif isinstance(mask_b, torch.Tensor):
                mask_b_np = mask_b.cpu().numpy()
            elif isinstance(mask_b, np.ndarray):
                mask_b_np = mask_b
            else:
                raise TypeError(f"mask_b deve essere numpy array, torch tensor o PIL Image, ma è {type(mask_b)}")

            if any(np.any(mask_b_np == cls) for cls in self.rare_classes):
                break

        # Resize immagini e maschere a crop_size
        target_size = (self.crop_size[0], self.crop_size[1])
        image = F.resize(image, target_size)
        mask = F.resize(mask, target_size, interpolation=InterpolationMode.NEAREST)

        if not isinstance(image_b, Image.Image):
            if isinstance(image_b, torch.Tensor):
                image_b = Image.fromarray((image_b.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            elif isinstance(image_b, np.ndarray):
                image_b = Image.fromarray(image_b.astype(np.uint8))
        image_b = F.resize(image_b, target_size)
        
        if not isinstance(mask_b, Image.Image):
            mask_b = Image.fromarray(mask_b_np.astype(np.uint8))
        mask_b = F.resize(mask_b, target_size, interpolation=InterpolationMode.NEAREST)

        mask_b_np = np.array(mask_b)
        class_mask = np.zeros_like(mask_b_np, dtype=np.uint8)
        for cls in self.rare_classes:
            class_mask[mask_b_np == cls] = 1
        class_mask = torch.from_numpy(class_mask).float().unsqueeze(0)

        # Converti in tensor e normalizza (immagini)
        image = to_tensor(image)
        image_b = to_tensor(image_b)
        image = normalize(image, self.mean, self.std)
        image_b = normalize(image_b, self.mean, self.std)

        mixed_image = image * (1 - class_mask) + image_b * class_mask

        # Maschere come tensori
        mask = torch.from_numpy(np.array(mask)).long()
        mask_b = torch.from_numpy(mask_b_np).long()
        mixed_mask = mask * (1 - class_mask.squeeze(0).long()) + mask_b * class_mask.squeeze(0).long()

        return mixed_image, mixed_mask


def val_transform_fn(image: Image.Image, mask: Image.Image):
    target_size = (512, 1024)

    # Resize immagine (height, width) --> PIL vuole (width, height)
    image = F.resize(image, target_size[::-1])
    image = F.to_tensor(image)
    image = F.normalize(image, mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    # Resize mask con interpolazione nearest
    mask = F.resize(mask, target_size[::-1], interpolation=InterpolationMode.NEAREST)
    mask = PILToTensor()(mask).long()
    mask = mask.squeeze(0) # Rimuove il canale extra se presente, shape finale: [H, W]

    return image, mask

def val_transform_fn_no_mask(image: Image.Image):
    target_size = (512, 1024)

    # Resize immagine (height, width) --> PIL vuole (width, height)
    image = F.resize(image, target_size[::-1])
    image = F.to_tensor(image)
    image = F.normalize(image, mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    # Resize mask con interpolazione nearest
    '''mask = F.resize(mask, target_size[::-1], interpolation=InterpolationMode.NEAREST)
    mask = PILToTensor()(mask).long()
    mask = mask.squeeze(0) # Rimuove il canale extra se presente, shape finale: [H, W]'''

    return image
