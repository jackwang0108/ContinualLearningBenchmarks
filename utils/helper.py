# Standard Library
import sys
import math
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
from loguru._defaults import LOGURU_FORMAT

# Torch Library
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid


def get_logger(log_file: Path, with_time: bool = True):
    global logger

    logger.remove()
    logger.add(log_file, level="DEBUG",
               format=f"{'{time:YYYY-D-MMMM@HH:mm:ss}' if with_time else ''}│ {{message}}")
    logger.add(sys.stderr, level="DEBUG",
               format=f"{'{time:YYYY-D-MMMM@HH:mm:ss}' if with_time else ''}│ <level>{{message}}</level>")

    return logger


def to_onehot(input: torch.FloatTensor, num_classes: int) -> torch.FloatTensor:
    ori_size = input.size()
    return F.one_hot(input.flatten(), num_classes=num_classes).reshape(*ori_size, -1)


def get_probas(logits: torch.FloatTensor) -> torch.FloatTensor:
    return F.softmax(logits, dim=-1)


def get_pred(probas: torch.FloatTensor) -> torch.FloatTensor:
    return probas.argmax(dim=-1)


def _reduction(array: np.ndarray, reduction: Literal["mean", "none"] = "mean") -> float | np.ndarray:
    if reduction == "mean":
        return array.mean()
    elif reduction == "none":
        return array
    else:
        raise NotImplementedError(f"{reduction} reduction is not supported")


def get_backward_transfer(cl_matrix: np.ndarray, reduction: Literal["mean", "none"] = "mean") -> float | np.ndarray:

    R_ii = np.diag(cl_matrix)[:-1]
    R_iN = cl_matrix[:-1, -1]

    return _reduction(R_iN - R_ii, reduction)


def get_forward_transfer(unlearned_perf: np.ndarray, cl_matrix: np.ndarray, reduction: Literal["mean", "none"] = "mean") -> float | np.ndarray:
    # TODO: finish this
    raise NotImplementedError


def get_last_setp_accuracy(cl_matrix: np.ndarray, reduction: Literal["mean", "none"] = "mean") -> float | np.ndarray:
    R_iN = cl_matrix[:, -1]
    return _reduction(R_iN, reduction)


def get_average_incremental_accuracy(cl_matrix: np.ndarray, reduction: Literal["mean", "none"] = "mean") -> float | np.ndarray:
    last_step_accuracies = [
        get_last_setp_accuracy(cl_matrix[:i, :i], "mean")
        for i in range(1, cl_matrix.shape[0])
    ]

    return _reduction(np.array(last_step_accuracies), reduction)


def get_forgetting_rate(cl_matrix: np.ndarray, reduction: Literal["mean", "none"] = "mean") -> float | np.ndarray:
    cl_matrix = cl_matrix[:-1, :]

    R_iN = cl_matrix[:, -1]
    R_i = np.max(cl_matrix[:, :-1], axis=1)

    return _reduction(R_i - R_iN, reduction)


@torch.no_grad()
def get_top1_acc(pred: torch.FloatTensor, gt: torch.FloatTensor, num_cls: int) -> torch.FloatTensor:
    gt = to_onehot(gt, num_cls)
    pred = to_onehot(pred, num_cls)
    correct = pred * gt
    return ((correct) != 0).sum() / pred.size(0)


def plot_matrix(cl_matrix: np.ndarray, current_task_id: int) -> Figure:
    mask = np.zeros_like(cl_matrix, dtype=bool)
    mask[np.tril_indices_from(cl_matrix, k=-1)] = True

    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(9, 9))

    # color map
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # create annotation
    annotation = [
        ["-" for _ in range(cl_matrix.shape[1])] for _ in range(cl_matrix.shape[0])]
    for i, j in itertools.product(range(cl_matrix.shape[0]), range(cl_matrix.shape[1])):
        if i <= j <= current_task_id:
            annotation[i][j] = f"{cl_matrix[i, j]:.2f}"

    sns_plot = sns.heatmap(
        cl_matrix, annot=annotation, mask=mask, cmap=cmap, vmin=0, vmax=1, fmt="",
        square=True, linewidths=.5, cbar_kws={"shrink": .5},
        annot_kws={"fontsize": 8},
        yticklabels=[f"Prev.Task {i}" for i in range(cl_matrix.shape[0])],
        xticklabels=[f"Curr.Task {i}" for i in range(cl_matrix.shape[1])]
    )

    return sns_plot.get_figure()


def draw_image(image: torch.FloatTensor, save_path: str | Path):
    grid_image = make_grid(image.clone().cpu(), nrow=int(
        math.sqrt(int(image.size(0)))))
    plt.imshow(grid_image.permute(1, 2, 0))
    plt.axis("off")
    plt.savefig(str(save_path))


if __name__ == "__main__":
    # images = torch.randn(16, 3, 32, 32)
    # draw_image(images, "./test.png")

    cl_matrix = np.random.random((10, 10))

    plot_matrix(cl_matrix, 2)
    plt.savefig("./example.png")

    # print(get_backward_transfer(cl_matrix))
    # print(get_forgetting_rate(cl_matrix))
