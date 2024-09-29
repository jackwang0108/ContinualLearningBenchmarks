# Standard Library
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
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter

# My Library
from model.iCaRL import iCaRL
from utils.helper import to_khot
from utils.annotation import (
    Task,
    Images,
    Labels,
    TaskLearner,
    PerformanceFunc,
)


device: torch.device = None
total_exemplar_size: int = None
exemplar_sets: dict[
    str, tuple[tuple[np.ndarray, torch.FloatTensor], Labels, torch.FloatTensor]
] = None


def get_model(backbone: nn.Module, module_args: Namespace) -> iCaRL:
    return iCaRL(backbone)


def get_args(argument_list: list[str]) -> tuple[Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Train iCaRL")

    parser.add_argument(
        "--buffer_size",
        type=int,
        default=2000,
        help="size of buffer, i.e. exemplar size",
    )
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
def get_preds(cl_model: iCaRL, images: torch.FloatTensor):

    # [B, 512]
    features: torch.FloatTensor = cl_model.feature_extractor(images)

    # Note: iCaRL L2-normalizes the features
    # ref: Page 2, Architecture, Paragraph 1, Line 5: "All feature vectors are L2-normalized..."
    features = features / features.norm(p=2, dim=1, keepdim=True)

    # [B, 1, 512]
    features = features.unsqueeze(dim=1)

    # [1, N, 512]
    cls_means: torch.FloatTensor = torch.cat(
        [exemplar_set[-1] for exemplar_set in exemplar_sets.values()], dim=0
    ).unsqueeze(dim=0)

    # [B, N, 512]
    pred = features - cls_means
    # [B, N]
    pred = pred.norm(p=2, dim=2, keepdim=False)
    return pred.argmin(dim=1)


def prepare_continual_learning(module_args: Namespace, **kwargs):
    global total_exemplar_size, exemplar_sets

    exemplar_sets = {}
    total_exemplar_size = module_args.buffer_size


def prepare_new_task(
    module_args: Namespace,
    **kwargs: dict[str, Any],
):
    global exemplar_sets

    train_dataset: Dataset = kwargs["train_dataset"]

    # merge exemplar set into current training dataset
    exemplar_images: Images
    exemplar_labels: Labels
    for exemplar_images, exemplar_labels, _ in exemplar_sets.values():
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
    for iCaRL, after learned each task, we need to manage the exemplar set, i.e., buffer management
    """
    global device, exemplar_sets, total_exemplar_size

    logger = kwargs["logger"]
    task_id = kwargs["task_id"]
    cl_model: iCaRL = kwargs["cl_model"]
    current_task: Task = kwargs["current_task"]
    test_dataset: Dataset = kwargs["test_dataset"]
    train_dataset: Dataset = kwargs["train_dataset"]
    cls_id_mapper: dict[str, int] = kwargs["cls_id_mapper"]

    # reduce existing exemplar sets
    if task_id > 0:
        logger.success("\treducing existing exemplar sets")

    exemplar_size = total_exemplar_size // len(cl_model.learned_classes)
    for cls_name in exemplar_sets:
        # 0 is image, 1 is label, 2 is mean feature
        reduced_image = exemplar_sets[cls_name][0]
        reduced_label = exemplar_sets[cls_name][1][:exemplar_size]

        # 0 is original image, 1 is augmented image
        reduced_original_image = reduced_image[0][:exemplar_size]
        reduced_augmented_image = reduced_image[1][:exemplar_size]

        exemplar_sets[cls_name] = (
            (reduced_original_image, reduced_augmented_image),
            reduced_label,
            # note: after reducing the exemplar set, the cls mean need to be recalculated
            calculate_features(
                cl_model,
                reduced_augmented_image.to(device=device),
            ).mean(dim=0, keepdim=True),
        )

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
            cl_model, labels, original_image, augmented_image, exemplar_size
        )

        exemplar_sets[cls_name] = (
            # save the original images, so denorm the image
            (exemplar_original_image, exemplar_augmented_image),
            exemplar_label,
            calculate_features(cl_model, exemplar_augmented_image).mean(
                dim=0, keepdim=True
            ),
        )

        logger.info(f"\t\t{cls_id=}, {cls_name=}")


@torch.no_grad()
def calculate_features(cl_model: iCaRL, images: torch.FloatTensor) -> torch.FloatTensor:

    images: torch.FloatTensor
    features: torch.FloatTensor
    features = cl_model.feature_extractor(images)
    features = features / features.norm(p=2, dim=1, keepdim=True)

    return features


@torch.no_grad()
def build_exemplar_set(
    cl_model: iCaRL,
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


def my_train_epoch(
    cl_model: iCaRL,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
) -> float:
    global device
    total_loss = 0

    sigmoid = nn.Sigmoid()
    distill_loss_func = nn.BCELoss(reduction="mean")
    classify_loss_func = nn.CrossEntropyLoss(reduction="mean")

    total_num_classes = len(cl_model.learned_classes) + len(cl_model.current_task)

    loss: torch.LongTensor
    original_image: torch.FloatTensor
    augmented_image: torch.FloatTensor
    label: torch.LongTensor
    for augmented_image, original_image, label in train_loader:
        augmented_image = augmented_image.to(device)
        label = to_khot(label, total_num_classes).to(device)

        # for new classes, use cross entropy loss
        logits = cl_model(augmented_image)
        classification_loss = classify_loss_func(logits, label)

        # for learned classes, use distillation loss, be cautious for the first task, there are no learned classes
        distillation_loss = torch.zeros_like(classification_loss)
        if len(cl_model.learned_classes) > 1:

            # get the logits of last model, i.e., teacher logits
            with torch.no_grad():
                with cl_model.use_previous_model():
                    # [B, Number of Learned Classes]
                    teacher_logits = cl_model(augmented_image)

            # get the logits of current model, i.e., student logits
            # [B, Number of Learned Classes + Number of Current Classes]
            student_logits = cl_model(augmented_image)

            # Note: iCaRL gives the logits as below, this mainly change the output logits to probability distribution
            # ref: Page 2, Architecture, Paragraph 2, Line 8: "The resulting network outputs are..."
            teacher_logits = sigmoid(teacher_logits)
            student_logits = sigmoid(student_logits)

            # get knowledge distillation loss
            # note, the output shape of teach logits and student logits are not the same, so alignment is necessary
            # The original paper just use the learned class logits of student logits to do distillation
            # ref: Page 3, Algorithm 3, \sum_{y=1}^{s-1}..., here y=1~s-1 means distill on learned class logits
            distillation_loss = distill_loss_func(
                student_logits[:, : len(cl_model.learned_classes)], teacher_logits
            )

        loss = classification_loss + distillation_loss

        total_loss += loss.clone().detach()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(train_loader)


def train_epoch(
    cl_model: iCaRL,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
) -> float:
    """
    This is the training code of original paper, which uses BCE loss
    """
    total_loss = 0

    sigmoid = nn.Sigmoid()
    loss_func = nn.BCEWithLogitsLoss(reduction="mean")

    num_new_classes = len(cl_model.current_task)
    num_learned_classes = len(cl_model.learned_classes)
    total_num_classes = num_new_classes + num_learned_classes

    loss: torch.FloatTensor
    target: torch.LongTensor
    original_image: torch.FloatTensor
    augmented_image: torch.FloatTensor

    for augmented_image, original_image, label in train_loader:
        augmented_image = augmented_image.to(device)
        target = to_khot(label, total_num_classes).to(device)

        # get the logits of current model
        logits = cl_model(augmented_image)

        # get logits of previous model for learned classes, be cautious for the first task, there are no learned classes
        if len(cl_model.learned_classes) > 0:

            # get the logits of last model, i.e., teacher logits
            with torch.no_grad():
                # [B, Number of Learned Classes]
                teacher_logits = cl_model.previous_nets[-1](augmented_image)

                # Note: iCaRL gives the logits as below, this mainly change the output logits to probability distribution
                # ref: Page 2, Architecture, Paragraph 2, Line 8: "The resulting network outputs are..."
                teacher_logits = sigmoid(teacher_logits)

            # get knowledge distillation loss
            # note, the output shape of teach logits and student logits are not the same, so alignment is necessary
            # The original paper just use the learned class logits of student logits to do distillation
            # ref: Page 3, Algorithm 3, \sum_{y=1}^{s-1}..., here y=1~s-1 means distill on learned class logits
            target = torch.cat(
                (
                    teacher_logits[:, :num_learned_classes],
                    target[:, num_learned_classes:total_num_classes],
                ),
                dim=1,
            )

        loss = loss_func(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.clone().detach()

    return total_loss / len(train_loader)


@torch.no_grad()
def test_epoch(
    cl_model: iCaRL,
    test_loader: DataLoader,
    perf_func: PerformanceFunc,
    num_total_class: int,
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    performance = []

    for augmented_image, original_image, label in test_loader:
        augmented_image, label = augmented_image.to(device), label.to(device)

        # iCaRL use Nearest-Mean-of-Exemplar to classify, so use that here
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
        cl_model: iCaRL = kwargs["cl_model"]
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
            milestones=[int(num_epoch * 7 / 10), int(num_epoch * 9 / 10)],
            gamma=0.2,
        )

        for epoch in range(num_epoch):

            train_loss = train_epoch(cl_model, train_loader, optimizer)

            # Note, when learning the current task, top1 acc cannot be calculated
            # because the exemplar sets of current task has not been built

            test_top1_acc = torch.nan
            # test_top1_acc = test_epoch(
            #     cl_model,
            #     test_loader,
            #     get_top1_acc,
            #     len(cl_model.learned_classes) + len(current_task),
            # )

            scheduler.step()

            # log epoch
            print_interval = num_epoch // main_args.log_times
            if (epoch + 1) % (print_interval if print_interval != 0 else 1) == 0:
                logger.info(
                    f"\tEpoch [{num_epoch}/{epoch+1:>{len(str(num_epoch))}d}], {train_loss=:.3f}, test_top1_acc not available for iCaRL when learning current task"
                )

            # log global
            training_watcher["Train Loss"] = train_loss

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
