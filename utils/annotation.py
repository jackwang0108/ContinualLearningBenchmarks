# Standard Library
from collections.abc import Callable
from typing import Literal, TypeVar, Optional

# Third-Party Library
import numpy as np

# Torch Library
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


Task = list[str]

Images = np.ndarray | torch.FloatTensor
Labels = np.ndarray | torch.LongTensor


ClassDataGetter = Callable[[str, int], tuple[Images, Labels]]

TaskDataGetter = Callable[[Task, list[int]], tuple[Images, Labels]]

PerformanceFunc = Callable[
    [torch.FloatTensor, torch.FloatTensor, int], torch.FloatTensor
]

TaskLearner = Callable[..., nn.Module]

CLAbilityTester = Callable[
    [int, Task, list[Task], list[DataLoader], nn.Module],
    tuple[np.ndarray, Optional[dict[str, float]]],
]

CollateFunc = Callable[
    [list[tuple[torch.FloatTensor, np.ndarray, np.int64]]],
    tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor],
]

MetricFunc = Callable[[np.ndarray, Literal["mean", "none"]], float | np.ndarray]
