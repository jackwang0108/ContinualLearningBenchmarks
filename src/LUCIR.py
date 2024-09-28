# Standard Library
import math
import argparse
from typing import Any
from argparse import Namespace

# Third-Party Library
import numpy as np
from loguru._logger import Logger

# Torch Library
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter

# My Library
from model.LUCIR import LUCIR
from utils.helper import (
    to_khot,
    get_top1_acc,
)
from utils.annotation import (
    Task,
    Images,
    Labels,
    TaskLearner,
    PerformanceFunc,
)


K: int = None
m: float = None
lambda_base: float = None
device: torch.device = None
memory_buffers: dict[str, tuple[tuple[np.ndarray, torch.FloatTensor], Labels]] = None


def get_model(backbone: nn.Module) -> LUCIR:
    return LUCIR(backbone)


def get_args(argument_list: list[str]) -> tuple[Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Train LUCIR")

    parser.add_argument(
        "--num_example",
        type=int,
        default=20,
        help="number of examples store in the memory buffer of a given class",
    )
    parser.add_argument(
        "--lambda_base",
        type=float,
        default=5,
        help="value of \\lambda_{base}",
    )
    parser.add_argument(
        "--m",
        type=float,
        default=5,
        help="value of margin threshold",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=2,
        help="number of top-K new class embeddings as hard negative",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="SGD",
        choices=["Adam", "SGD"],
        help="which optimizer to use",
    )
    model_args, unknow_args = parser.parse_known_args(argument_list)

    return model_args, unknow_args


@torch.no_grad()
def get_preds(cl_model: LUCIR, images: torch.FloatTensor):
    logits: torch.FloatTensor
    scaled_logits: torch.FloatTensor
    normalized_features, logits, scaled_logits = cl_model(images)
    probas = scaled_logits.softmax(dim=-1)
    preds = probas.argmax(dim=-1)
    return preds


def prepare_continual_learning(module_args: Namespace, **kwargs):
    global memory_buffers

    memory_buffers = {}


def prepare_new_task(
    module_args: Namespace,
    **kwargs: dict[str, Any],
):
    global memory_buffers

    train_dataset: Dataset = kwargs["train_dataset"]

    # merge exemplar set into current training dataset
    exemplar_images: Images
    exemplar_labels: Labels
    for exemplar_images, exemplar_labels in memory_buffers.values():
        current_task_labels: Labels = train_dataset.labels
        current_task_images: Images = train_dataset.images

        exemplar_original_images: torch.FloatTensor = exemplar_images[0]
        exemplar_augmented_images: torch.FloatTensor = exemplar_images[1]

        train_dataset.images = np.concatenate(
            (current_task_images, exemplar_original_images), axis=0
        )
        train_dataset.labels = np.concatenate(
            (current_task_labels, exemplar_labels), axis=0
        )


def finish_new_task(
    module_args: Namespace,
    **kwargs: dict[str, Any],
):
    """
    for LUCIR, after learned each task, we need to build memory buffer of new classes
    """
    global device, memory_buffers

    logger = kwargs["logger"]
    task_id = kwargs["task_id"]
    cl_model: LUCIR = kwargs["cl_model"]
    current_task: Task = kwargs["current_task"]
    test_dataset: Dataset = kwargs["test_dataset"]
    train_dataset: Dataset = kwargs["train_dataset"]
    cls_id_mapper: dict[str, int] = kwargs["cls_id_mapper"]

    num_example: int = module_args.num_example

    # build exemplar sets for new classes
    logger.success(f"\tbuilding exemplar sets for {task_id=}")

    train_images: np.ndarray = train_dataset.images
    train_labels: np.ndarray = train_dataset.labels
    for cls_name in current_task:
        cls_id = cls_id_mapper[cls_name]

        # get images and labels of this class
        cls_data_index = train_labels == cls_id

        labels: np.ndarray = train_labels[cls_data_index]
        original_image: np.ndarray = train_images[cls_data_index]
        augmented_image: torch.FloatTensor = torch.stack(
            # normalize the data
            [
                test_dataset.augment(original_image[i])
                for i in range(original_image.shape[0])
            ],
            dim=0,
        ).to(device=device)

        # get exemplar images and labels
        (
            exemplar_label,
            exemplar_original_image,
            exemplar_augmented_image,
        ) = build_exemplar_set(
            cl_model, labels, original_image, augmented_image, num_example
        )

        memory_buffers[cls_name] = (
            # save the original images, so denorm the image
            (exemplar_original_image, exemplar_augmented_image),
            exemplar_label,
        )

        logger.info(f"\t\t{cls_id=}, {cls_name=}")


@torch.no_grad()
def calculate_features(cl_model: LUCIR, images: torch.FloatTensor) -> torch.FloatTensor:

    images: torch.FloatTensor
    features: torch.FloatTensor
    features = cl_model.feature_extractor(images)
    features = features / features.norm(p=2, dim=1, keepdim=True)

    return features


@torch.no_grad()
def build_exemplar_set(
    cl_model: LUCIR,
    labels: torch.LongTensor,
    original_image: np.ndarray,
    augmented_image: torch.FloatTensor,
    exemplar_size: int,
) -> tuple[Images, Labels]:

    # get features
    # [N, 512]
    features = calculate_features(cl_model, augmented_image)

    # get mean of features
    # [1, 512]
    mean = features.mean(dim=0, keepdim=True)

    # build exemplar set
    exemplar_labels: list[np.ndarray] = []
    exemplar_features: list[torch.FloatTensor] = []
    exemplar_original_images: list[np.ndarray] = []
    exemplar_augmented_images: list[torch.FloatTensor] = []

    available_mask = torch.ones(features.size(0), dtype=bool)
    for k in range(1, exemplar_size + 1):
        # get k-th exemplar

        # [1, 512]
        current_sum = (
            torch.cat(exemplar_features, dim=0).sum(dim=0, keepdim=True)
            if exemplar_features
            else torch.zeros_like(mean)
        )
        # [N-k, 512]
        summation = mean - 1 / k * (features + current_sum)
        # [N-k]
        euclidean_distance = summation.norm(p=2, dim=1, keepdim=False)
        kth_exemplar_idx = euclidean_distance.argmin(dim=0).cpu().item()

        # add k-th exemplar into the exemplar set
        exemplar_labels.append(labels[kth_exemplar_idx])
        exemplar_features.append(features[kth_exemplar_idx].unsqueeze(dim=0))
        exemplar_original_images.append(original_image[kth_exemplar_idx])
        exemplar_augmented_images.append(augmented_image[kth_exemplar_idx])

        # remove k-th exemplar from cls_images/cls_labels/cls_features
        available_mask[kth_exemplar_idx] = False
        labels = labels[available_mask]
        features = features[available_mask]
        original_image = original_image[available_mask]
        augmented_image = augmented_image[available_mask]

        # reduce the available_mask, for 1 image has been erased
        available_mask = available_mask[available_mask]

    # return exemplar set
    return (
        np.array(exemplar_labels, dtype=np.int64),
        np.stack(exemplar_original_images, axis=0),
        torch.stack(exemplar_augmented_images, dim=0),
    )


def get_L_ce(
    scaled_logits: torch.FloatTensor, label: torch.LongTensor
) -> torch.FloatTensor:
    # label: [N, C], for one-hot label
    # scaled_logits: [N, C]
    cross_entropy = nn.CrossEntropyLoss()
    return cross_entropy(scaled_logits, label)


def get_LG_dis(
    new_features: torch.FloatTensor, old_features: torch.FloatTensor
) -> torch.FloatTensor:
    # label: [N, C], for one-hot label
    # scaled_logits: [N, C]
    cosine_similarity = nn.CosineEmbeddingLoss()
    return cosine_similarity(
        new_features,
        old_features,
        torch.ones(new_features.size(0), device=new_features.device),
    )


def get_L_mr(
    K: int,
    margin: float,
    num_old_classes: int,
    old_example_logits: torch.FloatTensor,
    old_example_onehot_label: torch.LongTensor,
):
    margin_ranking = nn.MarginRankingLoss(margin=margin)

    # get ground truth score, [B, 1]
    gt_score = (old_example_onehot_label * old_example_logits).sum(dim=-1)

    # get hard negative score, [B, K]
    hard_negative_score = old_example_logits[:, num_old_classes:].topk(k=K, dim=-1)[0]

    return sum(
        margin_ranking(
            gt_score,
            hard_negative_score[:, i],
            torch.ones(gt_score.size(0), device=gt_score.device),
        )
        for i in range(K)
    )


def train_epoch(
    cl_model: LUCIR,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
) -> float:
    """
    This is the training code of original paper, which uses BCE loss
    """
    total_loss = 0

    sigmoid = nn.Sigmoid()
    cross_entropy = nn.CrossEntropyLoss()

    num_new_classes = len(cl_model.current_task)
    num_learned_classes = len(cl_model.learned_classes)
    total_num_classes = num_new_classes + num_learned_classes

    loss: torch.FloatTensor
    label: torch.LongTensor
    onehot_label: torch.LongTensor
    original_image: torch.FloatTensor
    augmented_image: torch.FloatTensor

    for augmented_image, original_image, label in train_loader:
        augmented_image = augmented_image.to(device)
        onehot_label = to_khot(label, total_num_classes).to(device)

        # get the logits of current model
        normalized_features, logits, scaled_logits = cl_model(augmented_image)

        # get cross entropy loss
        loss_ce = get_L_ce(scaled_logits, onehot_label)

        # get the other losses
        loss_dis, loss_mr = 0, 0
        if len(cl_model.learned_tasks) > 0:

            # get geometry distillation loss
            with torch.no_grad():
                # [B, F]
                old_features = cl_model.previous_feature_extractors[-1](augmented_image)

                normalized_old_features = F.normalize(old_features, p=2, dim=-1)

            loss_dis = get_LG_dis(normalized_features, normalized_old_features)

            # get margin ranking loss
            old_example_idx = label < num_learned_classes
            if old_example_idx.any():
                old_example_logits = logits[old_example_idx]
                old_example_onehot_label = onehot_label[old_example_idx]

                loss_mr = get_L_mr(
                    K,
                    m,
                    num_learned_classes,
                    old_example_logits,
                    old_example_onehot_label,
                )

        cn = len(cl_model.current_task)
        co = 1 if len(cl_model.learned_tasks) == 0 else len(cl_model.learned_classes)
        lam = lambda_base * math.sqrt(cn / co)

        loss = loss_ce + lam * loss_dis + loss_mr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.clone().detach()

    return total_loss / len(train_loader)


@torch.no_grad()
def test_epoch(
    cl_model: LUCIR,
    test_loader: DataLoader,
    perf_func: PerformanceFunc,
    num_total_class: int,
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    performance = []

    for augmened_image, original_image, label in test_loader:
        augmened_image, label = augmened_image.to(device), label.to(device)

        preds = get_preds(cl_model, augmened_image)
        performance.append(perf_func(preds, label, num_total_class))

    return sum(performance) / len(performance)


def get_task_learner(
    main_args: Namespace, module_args: Namespace, **kwargs: dict[str, Any]
) -> TaskLearner:

    global m, K, lambda_base

    m = float(module_args.m)
    K = int(module_args.K)
    lambda_base = float(module_args.lambda_base)

    num_task_learned = 0

    def task_learner(**kwargs: dict[str, Any]) -> nn.Module:
        global device
        nonlocal main_args, module_args, num_task_learned

        task_id: int = kwargs["task_id"]
        current_task: Task = kwargs["current_task"]
        cl_model: LUCIR = kwargs["cl_model"]
        train_loader: DataLoader = kwargs["train_loader"]
        test_loader: DataLoader = kwargs["test_loader"]
        logger: Logger = kwargs["logger"]
        writer: SummaryWriter = kwargs["writer"]
        hparams_dict: dict = kwargs["hparams_dict"]
        training_watcher = kwargs["training_watcher"]

        device = next(cl_model.parameters()).device

        optimizer: Optimizer = getattr(optim, module_args.optim)(
            cl_model.parameters(), lr=float(main_args.lr), weight_decay=1e-5
        )

        num_epoch = main_args.epochs

        scheduler: LRScheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(num_epoch * 1 / 2), int(num_epoch * 3 / 4)],
            gamma=0.1,
        )

        for epoch in range(num_epoch):

            train_loss = train_epoch(cl_model, train_loader, optimizer)

            # Note, when learning the current task, top1 acc cannot be calculated
            # because the exemplar sets of current task has not been built

            test_top1_acc = test_epoch(
                cl_model,
                test_loader,
                get_top1_acc,
                len(cl_model.learned_classes) + len(current_task),
            )

            scheduler.step()

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

        # log hparams
        if num_task_learned == 0:
            hparams_dict["num_epoch"] = num_epoch
            hparams_dict["optim"] = optimizer.__class__.__name__
            hparams_dict["lr"] = optimizer.defaults["lr"]
            if (m := optimizer.defaults.get("momentum", None)) is not None:
                hparams_dict["momentum"] = m
            if (wd := optimizer.defaults.get("weight_decay", None)) is not None:
                hparams_dict["weight_decay"] = wd
            if (d := optimizer.defaults.get("dampening", None)) is not None:
                hparams_dict["dampening"] = d
        num_task_learned += 1

        return cl_model

    return task_learner
