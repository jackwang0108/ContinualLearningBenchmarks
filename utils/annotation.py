# Standard Library
from typing import Literal, TypeVar, Callable, Optional

# Third-Party Library
import numpy as np

# Torch Library
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


Task = list[str]

Images = np.ndarray
Labels = np.ndarray

Split = Literal["train", "val", "test"]

SupportedDataset = Literal["cifar100"]

# current supported dataset:
#   - data.cifar100.Cifar100Dataset
TorchDatasetImplementation = TypeVar("TorchDatasetImplementation")

ClassDataGetter = Callable[[str, int], tuple[Images, Labels]]

TaskDataGetter = Callable[[Task, list[int]], tuple[Images, Labels]]

PerformanceFunc = Callable[[torch.FloatTensor,
                            torch.FloatTensor, int], torch.FloatTensor]

TaskLearner = Callable[[int, int, nn.Module,
                        DataLoader, DataLoader], nn.Module]

CLAbilityTester = Callable[[int, list[list[str]], list[DataLoader],
                            nn.Module], tuple[np.ndarray, Optional[dict[str, float]]]]

MetricFunc = Callable[[np.ndarray,
                       Literal["mean", "none"]], float | np.ndarray]
