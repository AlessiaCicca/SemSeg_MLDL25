import torchvision.transforms.functional as F
from torchvision.transforms import RandomCrop, InterpolationMode, PILToTensor
import random
from PIL import Image
from torchvision.transforms.functional import resize, to_tensor, normalize, pil_to_tensor
from torchvision.transforms import InterpolationMode

class CustomAugmentation:
    def __init__(self, crop_size=(512, 1024), scale_choices=[0.75, 1.0, 1.5, 1.75, 2.0]):
        self.crop_size = crop_size
        self.scale_choices = scale_choices
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, image: Image.Image, mask: Image.Image):
        # Scaling
        scale = random.choice(self.scale_choices)
        new_size = (int(image.height * scale), int(image.width * scale))
        image = F.resize(image, new_size)
        mask = F.resize(mask, new_size, interpolation=InterpolationMode.NEAREST)

        # Flip
        if random.random() > 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)

        # Random crop
        i, j, h, w = RandomCrop.get_params(image, output_size=self.crop_size)
        image = F.crop(image, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)

        # Tensor & normalize
        image = F.to_tensor(image)
        image = F.normalize(image, mean=self.mean, std=self.std)
        mask = PILToTensor()(mask).long()

        return image, mask


def val_transform_fn(image, mask):
    target_size = (512, 1024)
    image = resize(image, target_size)
    image = to_tensor(image)
    image = normalize(image, mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    mask = resize(mask, target_size, interpolation=InterpolationMode.NEAREST)
    mask = pil_to_tensor(mask).long()

    return image, mask
