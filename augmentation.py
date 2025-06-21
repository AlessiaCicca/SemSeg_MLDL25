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
    """
    Randomly adjusts the gamma of an input image to alter its luminance contrast.

    Args:
        gamma : Maximum gamma adjustment factor.
                The applied gamma will be sampled from [1, 1 + gamma].

    """
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, img, mask):
        return adjust_gamma(img, random.uniform(1, 1 + self.gamma)), mask


class AdjustSaturation:
    """
    Randomly adjusts the saturation of an input image to vary color intensity.

    Args:
        saturation : Maximum saturation adjustment factor.
                    The applied saturation factor is sampled from 
                    [1 - saturation, 1 + saturation].
   
    """
    def __init__(self, saturation):
        self.saturation = saturation

    def __call__(self, img, mask):
        return adjust_saturation(img, random.uniform(1 - self.saturation, 1 + self.saturation)), mask


class AdjustHue:
    """
    Randomly adjusts the hue of an input image to alter its color tone.

    Args:
        hue : Maximum hue shift value.
              The applied hue shift is sampled from [-hue, hue].
    """
    def __init__(self, hue):
        self.hue = hue

    def __call__(self, img, mask):
        return adjust_hue(img, random.uniform(-self.hue, self.hue)), mask


class AdjustBrightness:
    """
    Randomly adjusts the brightness of an input image.

    Args:
        bf : Maximum brightness adjustment factor.
             The applied factor is sampled from [1 - bf, 1 + bf].
    """
    def __init__(self, bf):
        self.bf = bf

    def __call__(self, img, mask):
        return adjust_brightness(img, random.uniform(1 - self.bf, 1 + self.bf)), mask


class AdjustContrast:
    """
    Randomly adjusts the contrast of an input image.

    Args:
        cf : Maximum contrast adjustment factor.
             The applied factor is sampled from [1 - cf, 1 + cf].

    """
    def __init__(self, cf):
        self.cf = cf

    def __call__(self, img, mask):
        return adjust_contrast(img, random.uniform(1 - self.cf, 1 + self.cf)), mask


class RandomSized:
    """
    Randomly resizes the input image and mask by a scale factor between 0.5 and 2.0,
    then applies a random crop to the specified size.

    Args:
        size : Target crop size (square).

    """
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        w = int(random.uniform(0.5, 2) * img.size[0])
        h = int(random.uniform(0.5, 2) * img.size[1])
        img = img.resize((w, h), Image.BILINEAR)
        mask = mask.resize((w, h), Image.NEAREST)
        return RandomCrop(self.size)(img, mask)


class RandomCrop:
    """
    Randomly crops a region of the specified size from the input image and mask.
    If the image or mask is smaller than the target size, it is resized.

    Args:
        size : Target crop size as (height, width).

    """
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
    """
    Combines multiple data augmentation techniques for semantic segmentation tasks.
    
    This class applies a sequence of geometric and color augmentations, optionally
    including ClassMix augmentation. The augmentations are configurable through constructor
    arguments.

    Args:
        dataset : The dataset instance used for ClassMix augmentation.
        crop_size : The target size (height, width) for output images and masks.
        rare_classes : Class indices considered rare for ClassMix sampling.
        scale_choices : List of scale factors used for random resizing.
        use_flip : Whether to randomly apply horizontal flip.
        use_scale : Whether to randomly scale images.
        use_crop : Whether to randomly crop images.
        use_classmix : Whether to apply ClassMix blending using rare classes.
        use_brightness : Whether to randomly adjust image brightness.
        use_hue : Whether to randomly adjust image hue.
        use_gamma : Whether to randomly adjust image gamma.
        use_saturation : Whether to randomly adjust image saturation.
        use_contrast : Whether to randomly adjust image contrast.
        use_colorjitter : Whether to apply ColorJitter for additional color variations.

    The augmentations include:
        - Random scaling, flipping, rotation, cropping
        - Color transformations (gamma, brightness, contrast, saturation, hue)
        - ClassMix: blends regions containing rare classes from another image

    """
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

            
            if isinstance(image_b, torch.Tensor):
                image_b = Image.fromarray((image_b.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            if isinstance(mask_b, torch.Tensor):
                mask_b_np = mask_b.squeeze().cpu().numpy()
                mask_b = Image.fromarray(mask_b_np.astype(np.uint8))
            else:
                mask_b_np = np.array(mask_b)

            if any(np.any(mask_b_np == cls) for cls in self.rare_classes):
                break

        image_b = resize(image_b, target_size[::-1])
        mask_b = resize(mask_b, target_size[::-1], interpolation=InterpolationMode.NEAREST)

        mask_np = np.array(mask)
        mask_b_np = np.array(mask_b)

        class_mask = np.isin(mask_b_np, self.rare_classes).astype(np.uint8)
        class_mask_tensor = torch.from_numpy(class_mask).float().unsqueeze(0)

        image = normalize(to_tensor(image), self.mean, self.std)
        image_b = normalize(to_tensor(image_b), self.mean, self.std)

        mixed_image = image * (1 - class_mask_tensor) + image_b * class_mask_tensor

        mask = torch.from_numpy(mask_np).long()
        mask_b = torch.from_numpy(mask_b_np).long()
        mixed_mask = mask * (1 - class_mask_tensor.squeeze(0).long()) + mask_b * class_mask_tensor.squeeze(0).long()

        return mixed_image, mixed_mask



def val_transform_fn(image: Image.Image, mask: Image.Image):
    """
    Preprocesses validation images and masks for evaluation.

    Args:
        image : Input RGB image.
        mask : Corresponding segmentation mask.

    Returns:
        tuple:
            image_tensor : Normalized image tensor.
            mask_tensor : Segmentation mask tensor.

    The function resizes both image and mask to (512, 1024), applies
    normalization to the image, and converts the mask to a tensor with appropriate type.

    """
    target_size = (512, 1024)
    image = resize(image, target_size[::-1])
    image = normalize(to_tensor(image), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    mask = resize(mask, target_size[::-1], interpolation=InterpolationMode.NEAREST)
    mask = PILToTensor()(mask).long().squeeze(0)
    return image, mask

def val_transform_fn_no_mask(image: Image.Image):
    """
    Preprocesses validation images without corresponding masks.

    Args:
        image : Input RGB image.

    Returns:
        torch.Tensor: Normalized image tensor.

    The function resizes the image to (512, 1024) and applies normalization.
    Suitable for validating models on unlabeled target-domain data.
    
    """
    target_size = (512, 1024)
    target_size = (512, 1024)
    image = resize(image, target_size[::-1])
    image = normalize(to_tensor(image), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return image


