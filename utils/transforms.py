# Standard Library
import random
from typing import Optional
from contextlib import nullcontext

# Third-Party Library
import numpy as np
import PIL.Image as Image

# Torch Library
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional

# My Library
from .annotation import SupportedDataset, Split


class SameRandomStateContext:
    def __enter__(self):
        self.random_state = random.getstate()

    def __exit__(self, exc_type, exc_value, traceback):
        random.setstate(self.random_state)


class SeedableRandomSquareCrop:
    """ Apply a random square crop to the input image with a seedable behavior. The random crop status could be fixed. """

    def __init__(self, size: int):
        self.size = size

    def __call__(self, img: torch.FloatTensor) -> torch.FloatTensor:
        c, h, w = img.shape[-3:]
        x, y = 0, 0
        if h > self.size:
            y = random.randint(0, h - self.size)
        if w > self.size:
            x = random.randint(0, w - self.size)
        return functional.crop(img, y, x, self.size, self.size)


def get_crop_transform(
    size: int,
    is_eval: bool,
    same_crop: bool,
) -> SeedableRandomSquareCrop | transforms.CenterCrop | transforms.RandomCrop:
    """
    Returns the appropriate crop transform based on the input parameters.

    Args:
        size: An integer specifying the size of the cropped image.
        is_eval: A boolean indicating if the transform is for evaluation.
        same_transform: A boolean indicating if the same random crop should be used for all frames in a clip.

    Returns:
        The selected crop transform based on the input parameters.
    """

    if is_eval:
        return transforms.CenterCrop(size)
    elif same_crop:
        return SeedableRandomSquareCrop(size)
    else:
        return transforms.RandomCrop(size, padding=4)


def get_rgb_transform(
    is_eval: bool,
) -> nn.Module:
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
            transforms.RandomErasing(),
            # this slows down training aaaaaa lot
            # transforms.RandomRotation(30, expand=False)
        ])

    return transforms.Compose(img_transforms)


# ImageNet Mean and Std
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]

# Cifar100 MEAN and Std
CIFAR100_STD: list[float] = [0.2023, 0.1994, 0.2010]
CIFAR100_MEAN: list[float] = [0.4914, 0.4822, 0.4465]


class UnifiedTransforms(nn.Module):
    def __init__(
        self,
        mean: list[float],
        std: list[float],
        is_eval: bool,
        use_crop: bool,
        same_crop: bool = True,
        crop_size: Optional[int] = 32,
    ) -> None:
        super().__init__()

        self.is_eval = is_eval
        self.crop_size = crop_size
        self.use_crop = use_crop
        self.same_crop = same_crop

        self.to_tensor = transforms.ToTensor()
        self.rgb_transform = get_rgb_transform(is_eval)
        self.crop_transform = get_crop_transform(
            size=crop_size, is_eval=is_eval, same_crop=same_crop)

        self.normalize = transforms.Normalize(mean=mean, std=std)

    def forward(self, image: np.ndarray | Image.Image) -> torch.FloatTensor:

        # convert PIL Image and np.ndarray image to tensor
        # Note: this will converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        # Note: the PIL Image need to belong to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or the numpy.ndarray has dtype = np.uint8
        image = self.to_tensor(image)

        # apply crop transform
        if self.use_crop:
            with SameRandomStateContext() if self.same_crop else nullcontext():
                image = self.crop_transform(image)

        # apply rgb transform to rgb image
        image = self.rgb_transform(image) if self.is_eval else image

        # normalize the image
        image = self.normalize(image)

        return image


def get_transforms(
    dataset: SupportedDataset,
    crop_size: Optional[int] = 32,
    same_crop: Optional[bool] = False,
) -> tuple[UnifiedTransforms, UnifiedTransforms]:

    if dataset == "cifar100":
        mean, std = CIFAR100_MEAN, CIFAR100_STD
    else:
        raise NotImplementedError

    use_crop = crop_size is not None

    # train transform
    train_transform = UnifiedTransforms(
        mean=mean, std=std, is_eval=False, use_crop=use_crop, same_crop=same_crop, crop_size=crop_size)

    # test transform
    test_transform = UnifiedTransforms(
        mean=mean, std=std, is_eval=True, use_crop=use_crop,
        same_crop=True, crop_size=crop_size)

    return train_transform, test_transform
