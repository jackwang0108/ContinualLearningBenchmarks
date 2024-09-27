# Standard Library
import os
import sys
import math
import random
import itertools
from pathlib import Path
from typing import Literal, Callable

# Third-Party Library
import numpy as np
import seaborn as sns
from loguru import logger
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Torch Library
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

# My Library
from .annotation import MetricFunc


def get_logger(log_file: Path, with_time: bool = True):
    global logger

    logger.remove()
    logger.add(
        log_file,
        level="DEBUG",
        format=f"{'{time:YYYY-D-MMMM@HH:mm:ss}' if with_time else ''}│ {{message}}",
    )
    logger.add(
        sys.stderr,
        level="DEBUG",
        format=f"{'{time:YYYY-D-MMMM@HH:mm:ss}' if with_time else ''}│ <level>{{message}}</level>",
    )

    return logger


def set_random_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def to_khot(index: torch.IntTensor, num_classes: int) -> torch.IntTensor:
    # process if index is 1-dimensional, i.e. [batch_size]
    if index.ndim == 1:
        index = index.unsqueeze(dim=1)

    khot = torch.zeros(index.size(0), num_classes, device=index.device)
    khot.scatter_(1, index, 1)
    return khot


def get_probas(logits: torch.FloatTensor) -> torch.FloatTensor:
    return F.softmax(logits, dim=-1)


def get_pred(probas: torch.FloatTensor) -> torch.FloatTensor:
    return probas.argmax(dim=-1)


class CLMetrics:
    def __init__(self, metrics: dict[str, MetricFunc]) -> None:
        self.metrics = metrics

    def __call__(
        self, cl_matrix: np.ndarray, reduction: Literal["mean", "none"] = "mean"
    ) -> dict[str, float | np.ndarray]:
        return {
            metric_name: metric_func(cl_matrix, reduction)
            for metric_name, metric_func in self.metrics.items()
        }


def _reduction(
    array: np.ndarray, reduction: Literal["mean", "none"] = "mean"
) -> float | np.ndarray:
    if reduction == "mean":
        return array.mean()
    elif reduction == "none":
        return array
    else:
        raise NotImplementedError(f"{reduction} reduction is not supported")


def get_backward_transfer(
    cl_matrix: np.ndarray, reduction: Literal["mean", "none"] = "mean"
) -> float | np.ndarray:
    cl_matrix = cl_matrix.copy()
    R_ii = np.diag(cl_matrix)[:-1]
    R_iN = cl_matrix[:-1, -1]
    return _reduction(R_iN - R_ii, reduction)


def get_forward_transfer(
    unlearned_perf: np.ndarray,
    cl_matrix: np.ndarray,
    reduction: Literal["mean", "none"] = "mean",
) -> float | np.ndarray:
    # TODO: finish this
    raise NotImplementedError


def get_last_setp_accuracy(
    cl_matrix: np.ndarray, reduction: Literal["mean", "none"] = "mean"
) -> float | np.ndarray:
    cl_matrix = cl_matrix.copy()
    R_iN = cl_matrix[:, -1]
    return _reduction(R_iN, reduction)


def get_average_incremental_accuracy(
    cl_matrix: np.ndarray, reduction: Literal["mean", "none"] = "mean"
) -> float | np.ndarray:
    cl_matrix = cl_matrix.copy()
    last_step_accuracies = [
        get_last_setp_accuracy(cl_matrix[:i, :i], "mean")
        for i in range(1, cl_matrix.shape[0])
    ]

    return _reduction(np.array(last_step_accuracies), reduction)


def get_forgetting_rate(
    cl_matrix: np.ndarray, reduction: Literal["mean", "none"] = "mean"
) -> float | np.ndarray:
    cl_matrix = cl_matrix.copy()
    cl_matrix = cl_matrix[:-1, :]
    R_iN = cl_matrix[:, -1]
    R_i = np.max(cl_matrix[:, :-1], axis=1)
    return _reduction(R_i - R_iN, reduction)


# TODO: 计算Recall, Precision等指标


@torch.no_grad()
def get_top1_acc(
    top1_pred: torch.FloatTensor, gt: torch.FloatTensor, num_cls: int
) -> torch.FloatTensor:
    gt = to_khot(gt, num_cls)
    top1_pred = to_khot(top1_pred, num_cls)
    correct = top1_pred * gt
    return ((correct) != 0).sum() / top1_pred.size(0)


@torch.no_grad()
def get_top5_acc(
    top5_pred: torch.FloatTensor, gt: torch.FloatTensor, num_cls: int
) -> torch.FloatTensor:
    return get_top1_acc(top5_pred, gt, num_cls)


def plot_matrix(cl_matrix: np.ndarray, current_task_id: int) -> Figure:
    cl_matrix = cl_matrix.copy()
    mask = np.zeros_like(cl_matrix, dtype=bool)
    mask[np.tril_indices_from(cl_matrix, k=-1)] = True

    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(9, 9))

    # color map
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # create annotation
    annotation = [
        ["-" for _ in range(cl_matrix.shape[1])] for _ in range(cl_matrix.shape[0])
    ]
    for i, j in itertools.product(range(cl_matrix.shape[0]), range(cl_matrix.shape[1])):
        if i <= j <= current_task_id:
            annotation[i][j] = f"{cl_matrix[i, j]:.2f}"

    sns_plot = sns.heatmap(
        cl_matrix,
        annot=annotation,
        mask=mask,
        cmap=cmap,
        vmin=0,
        vmax=1,
        fmt="",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        annot_kws={"fontsize": 8},
        yticklabels=[f"Prev.Task {i}" for i in range(cl_matrix.shape[0])],
        xticklabels=[f"Curr.Task {i}" for i in range(cl_matrix.shape[1])],
    )

    return sns_plot.get_figure()


def draw_image(image: torch.FloatTensor, save_path: str | Path):
    if not (save_path := Path(save_path)).parent.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)
    grid_image = make_grid(image.clone().cpu(), nrow=int(math.sqrt(int(image.size(0)))))
    plt.tight_layout()
    plt.imshow(grid_image.permute(1, 2, 0))
    plt.axis("off")
    plt.savefig(str(save_path.resolve()))


if __name__ == "__main__":
    # images = torch.randn(16, 3, 32, 32)
    # draw_image(images, "./test.png")

    cl_matrix = np.random.random((10, 10))

    # plot_matrix(cl_matrix, 2)
    # plt.savefig("./example.png")

    bwt = get_backward_transfer(cl_matrix)
    print(bwt)
    # print(get_forgetting_rate(cl_matrix))
