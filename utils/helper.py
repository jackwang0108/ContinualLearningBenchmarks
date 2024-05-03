# Standard Library
import math
from pathlib import Path
from typing import Optional

# Third-Party Library
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Torch Library
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid


def to_onehot(input: torch.FloatTensor, num_classes: int) -> torch.FloatTensor:
    ori_size = input.size()
    return F.one_hot(input.flatten(), num_classes=num_classes).reshape(*ori_size, -1)


def get_probas(logits: torch.FloatTensor) -> torch.FloatTensor:
    return F.softmax(logits, dim=-1)


def get_pred(probas: torch.FloatTensor) -> torch.FloatTensor:
    return probas.argmax(dim=-1)


@torch.no_grad()
def get_top1_acc(pred: torch.FloatTensor, gt: torch.FloatTensor, num_cls: int) -> torch.FloatTensor:
    gt = to_onehot(gt, num_cls)
    pred = to_onehot(pred, num_cls)
    correct = pred * gt
    return ((correct) != 0).sum() / pred.size(0)


def plot_matrix(matrix: np.ndarray) -> Figure:
    ax: Axes
    fig: Figure
    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(matrix)
    ax.set_xticks(np.arange(matrix.shape[1]), xticks := [
                  f"Task {i}" for i in range(matrix.shape[1])])
    ax.set_yticks(np.arange(matrix.shape[1]), yticks := [
                  f"Task {i}" for i in range(matrix.shape[0])])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    for i in range(len(yticks)):
        for j in range(len(xticks)):
            if i < j:
                break
            text = ax.text(i, j, f"{matrix[i, j]:.2f}", fontsize=12,
                           ha="center", va="center", color="w")
    fig.tight_layout()
    return fig


def draw_image(image: torch.FloatTensor, save_path: str | Path):
    grid_image = make_grid(image.clone().cpu(), nrow=int(
        math.sqrt(int(image.size(0)))))
    plt.imshow(grid_image.permute(1, 2, 0))
    plt.axis("off")
    plt.savefig(str(save_path))


if __name__ == "__main__":
    # matrix = np.random.random((10, 10))
    # plot_matrix(matrix)
    # plt.savefig("./example.png")
    images = torch.randn(16, 3, 32, 32)
    draw_image(images, "./test.png")
