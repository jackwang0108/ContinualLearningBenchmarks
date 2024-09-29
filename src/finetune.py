# Standard Library
import argparse
from pathlib import Path
from typing import Any, Optional
from argparse import Namespace

# Third-Party Library
import numpy as np
from loguru._logger import Logger

# Torch Library
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# My Library
from model.finetune import Finetune
from utils.helper import to_khot, get_top1_acc
from utils.annotation import (
    Task,
    Images,
    Labels,
    TaskLearner,
    PerformanceFunc,
)

device: torch.device = None


def get_model(backbone: nn.Module, module_args: Namespace) -> Finetune:
    return Finetune(backbone)


def get_args(argument_list: list[str]) -> tuple[Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Train Finetune")

    parser.add_argument(
        "--optim",
        type=str,
        default="SGD",
        choices=["Adam", "SGD"],
        help="which optimizer to use",
    )
    model_args, unknown_args = parser.parse_known_args(argument_list)

    return model_args, unknown_args


@torch.no_grad()
def get_preds(cl_model: Finetune, images: torch.FloatTensor) -> torch.LongTensor:
    # sourcery skip: inline-immediately-returned-variable
    logits: torch.FloatTensor = cl_model(images)
    probas = logits.softmax(dim=-1)
    preds = probas.argmax(dim=-1)
    return preds


def prepare_continual_learning(module_args: Namespace, **kwargs):
    return None


def prepare_new_task(
    module_args: Namespace,
    **kwargs: dict[str, Any],
):
    return None


def finish_new_task(
    module_args: Namespace,
    **kwargs: dict[str, Any],
):
    return None


def train_epoch(
    cl_model: Finetune,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
) -> torch.FloatTensor:
    total_loss = 0

    loss_func = nn.CrossEntropyLoss()

    num_new_classes = len(cl_model.current_task)
    num_learned_classes = len(cl_model.learned_classes)
    total_num_classes = num_new_classes + num_learned_classes

    loss: torch.FloatTensor
    label: torch.LongTensor
    original_image: torch.FloatTensor
    augmented_image: torch.FloatTensor
    for augmented_image, original_image, label in train_loader:
        augmented_image = augmented_image.to(device)
        label = label.to(device)

        logits = cl_model(augmented_image)
        loss = loss_func(logits, label)

        total_loss += loss.clone().detach()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(train_loader)


@torch.no_grad()
def test_epoch(
    cl_model: Finetune,
    test_loader: DataLoader,
    perf_func: PerformanceFunc,
    num_total_class: int,
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    performance = []

    label: torch.LongTensor
    original_image: torch.FloatTensor
    augmented_image: torch.FloatTensor
    for augmented_image, original_image, label in test_loader:
        augmented_image, label = augmented_image.to(device), label.to(device)
        preds = get_preds(cl_model, augmented_image)
        performance.append(perf_func(preds, label, num_total_class))

    return sum(performance) / len(performance)


def get_task_learner(
    main_args: Namespace, module_args: Namespace, **kwargs: dict[str, Any]
) -> TaskLearner:

    num_task_learned = 0

    def task_learner(**kwargs: dict[str, Any]) -> nn.Module:
        global device
        nonlocal main_args, module_args, num_task_learned

        task_id: int = kwargs["task_id"]
        current_task: Task = kwargs["current_task"]
        cl_model: Finetune = kwargs["cl_model"]
        train_loader: DataLoader = kwargs["train_loader"]
        test_loader: DataLoader = kwargs["test_loader"]
        logger: Logger = kwargs["logger"]
        writer: SummaryWriter = kwargs["writer"]
        hparams_dict: dict = kwargs["hparams_dict"]
        training_watcher = kwargs["training_watcher"]

        device = next(cl_model.parameters()).device

        optimizer: Optimizer = getattr(optim, module_args.optim)(
            cl_model.parameters(),
            lr=float(main_args.lr),
            weight_decay=1e-5,
            # momentum=0.9,
        )

        num_epoch = main_args.epochs

        for epoch in range(num_epoch):

            train_loss = train_epoch(cl_model, train_loader, optimizer)

            test_top1_acc = test_epoch(
                cl_model,
                test_loader,
                get_top1_acc,
                len(cl_model.learned_classes) + len(current_task),
            )

            # log epoch
            print_interval = num_epoch // main_args.log_times
            if (epoch + 1) % (print_interval if print_interval != 0 else 1) == 0:
                logger.info(
                    f"\tEpoch [{num_epoch}/{epoch+1:>{len(str(num_epoch))}d}], {train_loss=:.3f}, {test_top1_acc=:.2f}"
                )

            # log global
            training_watcher["Train Loss"] = train_loss
            training_watcher["Test Top1 Acc"] = test_top1_acc

            # watch training
            for watcher_name, watcher_value in training_watcher.items():
                writer.add_scalar(
                    f"Task Learning/{watcher_name}",
                    scalar_value=watcher_value,
                    global_step=epoch + task_id * num_epoch,
                )

        # log extra hparams in optimizer
        if (m := optimizer.defaults.get("momentum", None)) is not None:
            hparams_dict["momentum"] = m
        if (wd := optimizer.defaults.get("weight_decay", None)) is not None:
            hparams_dict["weight_decay"] = wd
        if (d := optimizer.defaults.get("dampening", None)) is not None:
            hparams_dict["dampening"] = d

        num_task_learned += 1

        return cl_model

    return task_learner
