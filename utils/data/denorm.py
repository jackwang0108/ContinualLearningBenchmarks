# Standard Library

# Third-Party Library

# Torch Library
import torch


class DeNormalize(object):
    def __init__(self, mean: tuple[float], std: tuple[float]):
        self.mean = mean
        self.std = std

    def __call__(self, image: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) or (B, C, H, W) to be de-normalized.
        Returns:
            Tensor: De-Normalized image.
        """
        mean = torch.tensor(self.mean).view(
            (1, 3, 1, 1) if image.ndim == 4 else (3, 1, 1)
        )

        std = torch.tensor(self.std).view(
            (1, 3, 1, 1) if image.ndim == 4 else (3, 1, 1)
        )

        image.mul_(std).add_(mean)
        return image
