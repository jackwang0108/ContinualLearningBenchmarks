# Standard Library
import random
from typing import Optional
from contextlib import nullcontext

# Third-Party Library

# Torch Library
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional


class SameRandomStateContext:
    def __enter__(self):
        self.random_state = random.getstate()

    def __exit__(self, exc_type, exc_value, traceback):
        random.setstate(self.random_state)


IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]


class ThreeCrop:
    """ Apply three crops to the input image along the width dimension. """

    def __init__(self, dim: int):
        self._dim = dim

    def __call__(self, img: torch.FloatTensor) -> torch.FloatTensor:
        c, h, w = img.shape[-3:]
        y = (h - self._dim) // 2
        dw = w - self._dim
        ret = [functional.crop(img, y, x, self._dim, self._dim)
               for x in (0, dw // 2, dw)]
        return torch.stack(ret)


class RandomGaussianNoise(nn.Module):
    """ Apply random Gaussian noise to the input image with a given probability and standard deviation.  """

    def __init__(self, p: float = 0.5, s: float = 0.1):
        super().__init__()
        self.p = p
        self.std = s ** 0.5

    def forward(self, img: torch.FloatTensor) -> torch.FloatTensor:
        v = torch.rand(1)[0]
        if v < self.p:
            img += torch.randn(img.shape, device=img.device) * self.std
        return img


class SeedableRandomSquareCrop:
    """ Apply a random square crop to the input image with a seedable behavior. The random crop status could be fixed. """

    def __init__(self, dim: int):
        self._dim = dim

    def __call__(self, img: torch.FloatTensor) -> torch.FloatTensor:
        c, h, w = img.shape[-3:]
        x, y = 0, 0
        if h > self._dim:
            y = random.randint(0, h - self._dim)
        if w > self._dim:
            x = random.randint(0, w - self._dim)
        return functional.crop(img, y, x, self._dim, self._dim)


def get_crop_transform(
    is_eval: bool,
    same_crop_transform: bool,
    multi_crop: bool = False,
    crop_dim: Optional[int] = None,
) -> ThreeCrop | SeedableRandomSquareCrop | transforms.CenterCrop | transforms.RandomCrop:
    """
    Returns the appropriate crop transform based on the input parameters.

    Args:
        is_eval: A boolean indicating if the transform is for evaluation.
        crop_dim: An integer specifying the dimensions of the crop.
        same_transform: A boolean indicating if the same random crop should be used for all frames in a clip.
        multi_crop: A boolean indicating if multiple crops are used (default is False).

    Returns:
        The selected crop transform based on the input parameters.
    """
    crop_transform = None
    if crop_dim is not None:
        if multi_crop:
            assert is_eval, "Using ThreeCrop is only allow in evaluation phase"
            crop_transform = ThreeCrop(crop_dim)
        elif is_eval:
            crop_transform = transforms.CenterCrop(crop_dim)
        elif same_crop_transform:
            crop_transform = SeedableRandomSquareCrop(crop_dim)
        else:
            crop_transform = transforms.RandomCrop(crop_dim)
    return crop_transform


def get_rgb_transform(
    is_eval: bool,
) -> torch.ScriptModule:
    """
    Returns a TorchScript transform on the RGB image.

    Args:
        is_eval: A boolean indicating if the transform is for evaluation.

    Returns:
        torch.ScriptModule: TorchScript module with RGB image transformations.
    """
    img_transforms = []

    # Transforms for training
    if not is_eval:
        img_transforms.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(nn.ModuleList(
                [transforms.ColorJitter(hue=0.2)]), p=0.25),
            transforms.RandomApply(nn.ModuleList(
                [transforms.ColorJitter(saturation=(0.7, 1.2))]), p=0.25),
            transforms.RandomApply(nn.ModuleList(
                [transforms.ColorJitter(brightness=(0.7, 1.2))]), p=0.25),
            transforms.RandomApply(nn.ModuleList(
                [transforms.ColorJitter(contrast=(0.7, 1.2))]), p=0.25),
            transforms.RandomApply(nn.ModuleList(
                [transforms.GaussianBlur(kernel_size=5)]), p=0.25),
        ])

    # Transforms for both training and evaluating
    img_transforms.append(transforms.Normalize(
        mean=IMAGENET_MEAN, std=IMAGENET_STD))

    return torch.jit.script(nn.Sequential(*img_transforms))


class UnifiedTransforms(nn.Module):
    def __init__(
        self,
        is_eval: bool,
        use_crop_transform: bool,
        same_crop_transform: bool = True,
        multi_crop: bool = False,
        crop_dim: Optional[int] = 224
    ) -> None:
        super().__init__()

        self.is_eval = is_eval
        self.crop_dim = crop_dim
        self.use_crop_transform = use_crop_transform
        self.same_crop_transform = same_crop_transform

        self.image_transform = get_rgb_transform(is_eval)
        self.crop_transform = get_crop_transform(
            is_eval, same_crop_transform, multi_crop, crop_dim)

    def forward(self, image: torch.FloatTensor) -> torch.FloatTensor:
        assert not self.use_crop_transform or image.size(-2) >= self.crop_dim and image.size(-1) >= self.crop_dim, \
            f"Image is too small to crop, required crop size ({self.crop_dim}, {
            self.crop_dim}) is larger than input image size {image.shape[-2:]}"

        with SameRandomStateContext() if self.same_crop_transform else nullcontext():
            image = self.crop_transform(
                image) if self.use_crop_transform else image

        image = self.image_transform(image) if self.is_eval else image

        return image
