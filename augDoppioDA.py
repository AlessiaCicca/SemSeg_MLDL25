import random
import numbers
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode, PILToTensor
from torchvision.transforms.functional import (
    to_tensor, normalize, resize, hflip, rotate,
    adjust_gamma, adjust_saturation, adjust_hue,
    adjust_brightness, adjust_contrast
)
class AdjustGamma:
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, img, mask):
        return adjust_gamma(img, random.uniform(1, 1 + self.gamma)), mask


class AdjustSaturation:
    def __init__(self, saturation):
        self.saturation = saturation

    def __call__(self, img, mask):
        return adjust_saturation(img, random.uniform(1 - self.saturation, 1 + self.saturation)), mask


class AdjustHue:
    def __init__(self, hue):
        self.hue = hue

    def __call__(self, img, mask):
        return adjust_hue(img, random.uniform(-self.hue, self.hue)), mask


class AdjustBrightness:
    def __init__(self, bf):
        self.bf = bf

    def __call__(self, img, mask):
        return adjust_brightness(img, random.uniform(1 - self.bf, 1 + self.bf)), mask


class AdjustContrast:
    def __init__(self, cf):
        self.cf = cf

    def __call__(self, img, mask):
        return adjust_contrast(img, random.uniform(1 - self.cf, 1 + self.cf)), mask


class RandomSized:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        w = int(random.uniform(0.5, 2) * img.size[0])
        h = int(random.uniform(0.5, 2) * img.size[1])
        img = img.resize((w, h), Image.BILINEAR)
        mask = mask.resize((w, h), Image.NEAREST)
        return RandomCrop(self.size)(img, mask)


class RandomCrop:
    def __init__(self, size):
        self.size = size if not isinstance(size, numbers.Number) else (int(size), int(size))

    def __call__(self, img, mask):
        w, h = img.size
        th, tw = self.size
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))

class CombinedAugmentation:
    def __init__(self, dataset, crop_size=(512, 1024), rare_classes=[6, 11, 17],
                 scale_choices=[0.75, 1.0, 1.5, 1.75, 2.0],
                 use_flip=False, use_scale=False, use_crop=False, use_classmix=False,
                 use_brightness=False, use_hue=False, use_gamma=False, use_saturation=False, use_contrast=False, use_colorjitter=True):

        self.dataset = dataset
        self.crop_size = crop_size
        self.rare_classes = rare_classes
        self.scale_choices = scale_choices

        self.use_flip = use_flip
        self.use_scale = use_scale
        self.use_crop = use_crop
        self.use_classmix = use_classmix
        self.use_brightness = use_brightness
        self.use_hue = use_hue
        self.use_gamma = use_gamma
        self.use_saturation = use_saturation
        self.use_contrast = use_contrast
        self.use_colorjitter = use_colorjitter

        self.color_jitter = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        )
        self.random_sized = RandomSized(crop_size[0])
        self.adjust_gamma = AdjustGamma(0.5)
        self.adjust_saturation = AdjustSaturation(0.5)
        self.adjust_hue = AdjustHue(0.1)
        self.adjust_brightness = AdjustBrightness(0.3)
        self.adjust_contrast = AdjustContrast(0.3)

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, image, mask):
        # Resize, Flip, Rotate
        if self.use_scale:
            scale = random.choice(self.scale_choices)
            new_size = (int(image.height * scale), int(image.width * scale))
            image = resize(image, new_size)
            mask = resize(mask, new_size, interpolation=InterpolationMode.NEAREST)

        if self.use_flip and random.random() > 0.5:
            image = hflip(image)
            mask = hflip(mask)

        angle = random.uniform(-10, 10)
        image = rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
        mask = rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

        if self.use_crop:
            image, mask = self.random_sized(image, mask)

        # Color transformations
        color_transforms = []
        if self.use_gamma:
            color_transforms.append(self.adjust_gamma)
        if self.use_saturation:
            color_transforms.append(self.adjust_saturation)
        if self.use_hue:
            color_transforms.append(self.adjust_hue)
        if self.use_brightness:
            color_transforms.append(self.adjust_brightness)
        if self.use_contrast:
            color_transforms.append(self.adjust_contrast)

        random.shuffle(color_transforms)
        for t in color_transforms:
            if random.random() < 0.5:
                image, mask = t(image, mask)


        if self.use_colorjitter and random.random() < 0.5:
            image = self.color_jitter(image)

        target_size = self.crop_size
        image = resize(image, target_size[::-1])
        mask = resize(mask, target_size[::-1], interpolation=InterpolationMode.NEAREST)

        if self.use_classmix and random.random() < 0.5:
            return self.apply_classmix(image, mask)
        else:
            image = normalize(to_tensor(image), self.mean, self.std)
            mask = torch.from_numpy(np.array(mask)).long()
            return image, mask

    def apply_classmix(self, image, mask):
        target_size = self.crop_size

        while True:
            idx = random.randint(0, len(self.dataset) - 1)
            image_b, mask_b = self.dataset[idx]

            # Convert to PIL if needed
            if isinstance(image_b, torch.Tensor):
                image_b = Image.fromarray((image_b.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            if isinstance(mask_b, torch.Tensor):
                mask_b_np = mask_b.squeeze().cpu().numpy()
                mask_b = Image.fromarray(mask_b_np.astype(np.uint8))
            else:
                mask_b_np = np.array(mask_b)

            # Check for rare classes
            if any(np.any(mask_b_np == cls) for cls in self.rare_classes):
                break

        # Resize second pair
        image_b = resize(image_b, target_size[::-1])
        mask_b = resize(mask_b, target_size[::-1], interpolation=InterpolationMode.NEAREST)

        mask_np = np.array(mask)
        mask_b_np = np.array(mask_b)

        # Create binary mask where rare classes appear
        class_mask = np.isin(mask_b_np, self.rare_classes).astype(np.uint8)
        class_mask_tensor = torch.from_numpy(class_mask).float().unsqueeze(0)

        # Convert images to tensor and normalize
        image = normalize(to_tensor(image), self.mean, self.std)
        image_b = normalize(to_tensor(image_b), self.mean, self.std)

        # Blend images and masks
        mixed_image = image * (1 - class_mask_tensor) + image_b * class_mask_tensor

        mask = torch.from_numpy(mask_np).long()
        mask_b = torch.from_numpy(mask_b_np).long()
        mixed_mask = mask * (1 - class_mask_tensor.squeeze(0).long()) + mask_b * class_mask_tensor.squeeze(0).long()

        return mixed_image, mixed_mask



def val_transform_fn(image: Image.Image, mask: Image.Image):
    target_size = (512, 1024)
    image = resize(image, target_size[::-1])
    image = normalize(to_tensor(image), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    mask = resize(mask, target_size[::-1], interpolation=InterpolationMode.NEAREST)
    mask = PILToTensor()(mask).long().squeeze(0)
    return image, mask

def val_transform_fn_no_mask(image: Image.Image):
    target_size = (512, 1024)
    target_size = (512, 1024)
    image = resize(image, target_size[::-1])
    image = normalize(to_tensor(image), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return image


