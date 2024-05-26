# Standard Library
import datetime
from pathlib import Path
from typing import Optional

# Third-Party Library
import numpy as np

# Torch Library
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter

# My Library
from model.LingoCL import iCaRL_LingoCL
from utils.helper import get_logger
from utils.datasets import (get_dataset,
                            get_cls_data_getter,
                            CLDatasetGetter)
from utils.transforms import get_transforms
from utils.data.cifar100 import Cifar100Dataset
from utils.helper import (plot_matrix, draw_image)
from utils.helper import (get_probas, get_pred, to_khot)
from utils.helper import (get_top1_acc,
                          get_backward_transfer,
                          get_last_setp_accuracy,
                          get_average_incremental_accuracy,
                          get_forgetting_rate,
                          CLMetrics)
from utils.annotation import SupportedDataset, Images, Labels
from utils.annotation import (
    TaskLearner,
    PerformanceFunc,
    CLAbilityTester,
)

rname = input("the name of this running: ")
wname = f"{rname}-{datetime.datetime.now().strftime('%m-%d %H.%M')}"
writer = SummaryWriter(log_dir := f"log/{wname}")
logger = get_logger(Path(log_dir) / "running.log")
device = torch.device("cuda:1")

hparams_dict = {}


crop_size = 32


def train_epoch(model: iCaRL_LingoCL, train_loader: DataLoader, loss_func: nn.Module, optimizer: optim.Optimizer) -> float:
    total_loss = 0

    sigmoid = nn.Sigmoid()
    distill_loss_func = nn.BCELoss(reduction="mean")
    classify_loss_func = nn.CrossEntropyLoss(reduction="mean")

    total_num_classes = len(model.learned_classes) + len(model.current_task)

    loss: torch.FloatTensor
    image: torch.FloatTensor
    label: torch.LongTensor
    for image, label in train_loader:
        image = image.to(device)
        label = to_khot(label, total_num_classes).to(device)

        # for new classes, use cross entropy loss
        logits = model(image)
        classification_loss = classify_loss_func(logits, label)

        # for learned classes, use distillation loss, be cautious for the first task, there are no learned classes
        distillation_loss = torch.zeros_like(classification_loss)
        if len(model.learned_classes) > 1:

            # get the logits of last model, i.e., teacher logits
            with torch.no_grad():
                with model.use_previous_model():
                    # [B, Number of Learned Classes]
                    teacher_logits = model(image)

            # get the logits of current model, i.e., student logits
            # [B, Number of Learned Classes + Number of Current Classes]
            student_logits = model(image)

            # Note: iCaRL gives the logits as below, this mainly change the output logits to probability distribution
            # ref: Page 2, Architecture, Paragraph 2, Line 8: "The resulting network outputs are..."
            teacher_logits = sigmoid(teacher_logits)
            student_logits = sigmoid(student_logits)

            # get knowledge distillation loss
            # note, the output shape of teach logits and student logits are not the same, so alignment is necessary
            # The original paper just use the learned class logits of student logits to do distillation
            # ref: Page 3, Algorithm 3, \sum_{y=1}^{s-1}..., here y=1~s-1 means distill on learned class logits
            distillation_loss = distill_loss_func(
                student_logits[:, :len(model.learned_classes)], teacher_logits)

        loss = classification_loss + distillation_loss

        total_loss += loss.clone().detach()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(train_loader)


def paper_train_epoch(model: iCaRL_LingoCL, train_loader: DataLoader, loss_func: nn.Module, optimizer: optim.Optimizer) -> float:
    """
    This is the training code of original paper, which uses BCE loss for both new classes and learned classes.

    This version of training actually worse than the training code above, for CrossEntropy loss excels BCE loss for new classes learning.

    After the training, new classes exemplar is generated using the model and is used to classify. So, since the CrossEntropy loss gets a
    better feature extractor, the above training code gets better results on all classes.
    """
    total_loss = 0

    sigmoid = nn.Sigmoid()
    loss_func = nn.BCELoss(reduction="mean")

    total_num_classes = len(model.learned_classes) + len(model.current_task)

    loss: torch.FloatTensor
    image: torch.FloatTensor
    label: torch.LongTensor
    for image, label in train_loader:
        image = image.to(device)
        label = to_khot(label, total_num_classes).to(device)

        # get the logits of current model
        logits = model(image)

        # get logits of previous model for learned classes, be cautious for the first task, there are no learned classes
        if len(model.learned_classes) > 0:

            # get the logits of last model, i.e., teacher logits
            with torch.no_grad():
                with model.use_previous_model():
                    # [B, Number of Learned Classes]
                    teacher_logits = model(image)

            # Note: iCaRL gives the logits as below, this mainly change the output logits to probability distribution
            # ref: Page 2, Architecture, Paragraph 2, Line 8: "The resulting network outputs are..."
            teacher_logits = sigmoid(teacher_logits)

            # get knowledge distillation loss
            # note, the output shape of teach logits and student logits are not the same, so alignment is necessary
            # The original paper just use the learned class logits of student logits to do distillation
            # ref: Page 3, Algorithm 3, \sum_{y=1}^{s-1}..., here y=1~s-1 means distill on learned class logits
            label[:, :len(model.learned_classes)] = teacher_logits

        loss = loss_func(sigmoid(logits), label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.clone().detach()

    return total_loss / len(train_loader)


@torch.no_grad()
def test_epoch(model: iCaRL_LingoCL, test_loader: DataLoader, perf_func: PerformanceFunc, num_cls_per_task: int) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    performance = []

    # Note: iCaRL use Nearest-Mean-of-Exemplars to classify, here using output logits just for monitor the learning
    for image, label in test_loader:
        image, label = image.to(device), label.to(device)
        preds = get_pred(get_probas(model(image)))
        performance.append(
            perf_func(preds, label, len(model.learned_classes) + num_cls_per_task))

    return sum(performance) / len(performance)


@torch.no_grad()
def calculate_cls_features(cls_data_loader: DataLoader, model: iCaRL_LingoCL) -> torch.FloatTensor:
    cls_features = []

    image: torch.FloatTensor
    features: torch.FloatTensor
    for image, _ in cls_data_loader:
        image = image.to(device)
        features = model.feature_extractor(image)
        features = features / features.norm(p=2, dim=1, keepdim=True)
        cls_features.append(features)

    return torch.cat(cls_features, dim=0)


@torch.no_grad()
def calculate_exemplar_mean(dataset: SupportedDataset, exemplar_image: Images, exemplar_label: Labels, model: iCaRL_LingoCL) -> torch.FloatTensor:
    # use test transforms here, for the predictions is made on test images
    _, transforms = get_transforms(
        dataset=dataset, crop_size=crop_size, same_crop=False)

    exemplar_dataset = get_dataset(dataset)(
        exemplar_image, exemplar_label, transforms)

    exemplar_loader = DataLoader(
        exemplar_dataset, batch_size=32, shuffle=False, num_workers=2)

    # [N, 512]
    cls_features = calculate_cls_features(exemplar_loader, model)

    return cls_features.mean(dim=0, keepdim=True)


@torch.no_grad()
def build_exemplar_set(dataset: SupportedDataset, cls_name: str, cls_id: int, exemplar_size: int, model: iCaRL_LingoCL) -> tuple[Images, Labels]:
    cls_data_getter = get_cls_data_getter(dataset, "train")

    cls_images, cls_labels = cls_data_getter(cls_name, cls_id)

    # use test transforms here, for the predictions is made on test images
    _, transforms = get_transforms(
        dataset=dataset, crop_size=crop_size, same_crop=False)

    cls_data_dataset = get_dataset(dataset)(cls_images, cls_labels, transforms)

    cls_data_loader = DataLoader(
        cls_data_dataset, batch_size=32, shuffle=False, num_workers=2)

    # get features
    # [N, 512]
    cls_features = calculate_cls_features(cls_data_loader, model)

    # get class mean
    # [1, 512]
    cls_mean = cls_features.mean(dim=0, keepdim=True)

    # build exemplar set
    exemplar_images: list[Images] = []
    exemplar_labels: list[int] = []
    exemplar_features: list[torch.FloatTensor] = []

    available_mask = torch.ones(cls_features.size(0), dtype=bool)
    for k in range(1, exemplar_size + 1):
        # get k-th exemplar

        # [1, 512]
        current_sum = (
            torch.cat(exemplar_features, dim=0).sum(dim=0, keepdim=True)
            if exemplar_features
            else torch.zeros_like(cls_mean)
        )
        # [N-k, 512]
        summation = cls_mean - 1 / k * \
            (cls_features + current_sum)
        # [N-k]
        norm = summation.norm(p=2, dim=1, keepdim=False)
        kth_exemplar_idx = norm.argmin(dim=0).cpu().item()

        # add k-th exemplar into the exemplar set
        exemplar_images.append(cls_images[kth_exemplar_idx])
        exemplar_labels.append(cls_labels[kth_exemplar_idx])
        exemplar_features.append(
            cls_features[kth_exemplar_idx].unsqueeze(dim=0))

        # remove k-th exemplar from cls_images/cls_labels/cls_features
        available_mask[kth_exemplar_idx] = False
        cls_features = cls_features[available_mask]
        cls_images = cls_images[available_mask]
        cls_labels = cls_labels[available_mask]

        # reduce the available_mask, for 1 image has been erased
        available_mask = available_mask[available_mask]

    # return exemplar set
    return np.stack(exemplar_images, axis=0), np.array(exemplar_labels, dtype=np.int64)


def get_task_learner() -> TaskLearner:
    num_task_learned = 0

    def task_learner(task_id: int, current_task: list[str], num_cls_per_task: int, model: iCaRL_LingoCL, train_loader: DataLoader, test_loader: DataLoader) -> iCaRL_LingoCL:

        loss_func = nn.CrossEntropyLoss()

        # Note: if use SGD, the performance is much more weaker than use Adam
        # Note: this may because of the original code uses gradient clip
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        # optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=5e-5)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[49, 63], gamma=0.2)

        log_times = 5
        num_epoch = 70
        for epoch in range(num_epoch):
            train_loss = train_epoch(model, train_loader, loss_func, optimizer)

            test_top1_acc = test_epoch(model, test_loader, get_top1_acc,
                                       num_cls_per_task) * 100

            scheduler.step()

            if (epoch + 1) % (num_epoch // log_times) == 0:
                logger.info(
                    f"\tEpoch [{num_epoch}/{epoch+1:>{len(str(num_epoch))}d}], {train_loss=:.3f}, {test_top1_acc=:.2f}")

            # log training
            training_watcher = {
                "Train Loss": train_loss,
                "Test Top1 Acc": test_top1_acc,
            }

            # watch training
            for watcher_name, watcher_value in training_watcher.items():
                writer.add_scalar(f"Task Learning/{watcher_name}",
                                  scalar_value=watcher_value, global_step=epoch + task_id * num_epoch)

        # log hparams
        nonlocal num_task_learned
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

        return model
    return task_learner


def get_continual_learning_ability_tester(task_num: int, num_cls_per_task: int) -> CLAbilityTester:

    num_cls_per_task = num_cls_per_task
    cl_matrix = np.zeros((task_num, task_num))

    metrics_getter = CLMetrics({
        "Backward Transfer": get_backward_transfer,
        "Forgetting Rate": get_forgetting_rate,
        "Last Step Accuracy": get_last_setp_accuracy,
        "Average Incremental Accuracy": get_average_incremental_accuracy,
    })

    @ torch.no_grad
    def continual_learning_ability_tester(task_id: int, learned_tasks: list[list[str]], learned_task_loaders: list[DataLoader], model: iCaRL_LingoCL) -> tuple[np.ndarray, dict[str, float]]:
        nonlocal cl_matrix, num_cls_per_task, metrics_getter

        # test on all tasks, including previous and current task
        for i, previous_loader in enumerate(learned_task_loaders):

            # iCaRL use Nearest-Mean-of-Exemplar to classify, so use that here
            performance = []

            image: torch.FloatTensor
            label: torch.FloatTensor
            for image, label in previous_loader:
                image, label = image.to(device), label.to(device)
                preds = model.get_preds(image)
                performance.append(
                    get_top1_acc(preds, label, len(model.learned_classes))
                )

            cl_matrix[i, task_id] = sum(performance) / len(performance)

            logger.info(
                f"\ttest on task {i}, test_acc={cl_matrix[i, task_id]: .2f}, {learned_tasks[i]}")

        # calculate continual learning ability metrics and log to summarywriter
        if task_id >= 1:
            current_cl_matrix = cl_matrix[:task_id+1, :task_id+1]

            current_metrics = metrics_getter(current_cl_matrix, "mean")

            # log to summarywriter
            for metric_name, metric_value in current_metrics.items():
                if isinstance(metric_value, (float, int)):
                    writer.add_scalar(
                        tag=f"Continual Learning Metrics/{metric_name}",
                        scalar_value=metric_value,
                        global_step=task_id
                    )

        # draw heatmap of cl_matrix and log to summarywriter
        writer.add_figure(
            tag=f"cl_matrix/{wname}",
            figure=plot_matrix(cl_matrix, task_id),
            global_step=task_id,
        )

        return cl_matrix, current_metrics if task_id >= 1 else {}

    return continual_learning_ability_tester


def continual_learning():

    dataset_getter = CLDatasetGetter(
        dataset="cifar100", task_num=10, fixed_task=False)

    model: iCaRL_LingoCL
    model = iCaRL_LingoCL().to(device)

    # iCaRL needs a exemplar set
    total_exemplar_size = 2000
    exemplar_set: dict[str, tuple[Images, Labels]] = {}

    # logging
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info("Task List:")
    for task_id, task in enumerate(dataset_getter.tasks):
        logger.info(f"\tTask {task_id}: {task}")

    # get task learner and cl-ability tester
    task_learner: TaskLearner = get_task_learner()

    continual_learning_ability_tester: CLAbilityTester = get_continual_learning_ability_tester(
        dataset_getter.task_num, dataset_getter.num_cls_per_task)

    train_dataset: Cifar100Dataset
    test_dataset: Cifar100Dataset
    learned_tasks: list[DataLoader] = []
    learned_task_loaders: list[DataLoader] = []
    for task_id, current_task, train_dataset, test_dataset in dataset_getter:
        # prepare the data for the task

        # merge exemplar set into current training dataset
        exemplar_images: Images
        exemplar_labels: Labels
        for exemplar_images, exemplar_labels in exemplar_set.values():
            current_task_images: Images = train_dataset.images
            current_task_labels: Labels = train_dataset.labels
            train_dataset.images = np.concatenate(
                (current_task_images, exemplar_images), axis=0)
            train_dataset.labels = np.concatenate(
                (current_task_labels, exemplar_labels), axis=0)

        # get dataloader
        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True, num_workers=2)
        test_loader = DataLoader(
            test_dataset, batch_size=32, shuffle=True, num_workers=2)

        # learn the new task
        with model.set_new_task(current_task):

            logger.success(f"{task_id=}, {current_task=}")
            model = task_learner(
                task_id, current_task, dataset_getter.num_cls_per_task, model, train_loader, test_loader)

        # exemplar set management
        # reduce existing exemplar sets
        exemplar_size = total_exemplar_size // len(model.learned_classes)
        for cls_name in exemplar_set:
            # 0 is image, 1 is label
            reduced_image = exemplar_set[cls_name][0][:exemplar_size]
            reduced_label = exemplar_set[cls_name][1][:exemplar_size]
            exemplar_set[cls_name] = (reduced_image, reduced_label)

            # note: after reducing the exemplar set, the cls mean need to be recalculated
            model.exemplar_means[cls_name] = calculate_exemplar_mean(
                "cifar100", reduced_image, reduced_label, model)

        # build exemplar sets for new classes
        for cls_name in current_task:
            cls_id = dataset_getter.cls_id_mapper[cls_name]

            # get exemplar images and labels
            exemplar_image, exemplar_label = build_exemplar_set(
                "cifar100", cls_name, cls_id, exemplar_size, model)

            exemplar_set[cls_name] = (exemplar_image, exemplar_label)

            # save exemplar set means to model
            model.exemplar_means[cls_name] = calculate_exemplar_mean(
                "cifar100", exemplar_image, exemplar_label, model)

        # save the test loader for continual learning testing
        learned_tasks.append(current_task)
        learned_task_loaders.append(test_loader)

        # test continual learning performance using standard metrics
        cl_matrix, metrics = continual_learning_ability_tester(
            task_id, learned_tasks, learned_task_loaders, model)

        if metrics is not None:
            logger.debug(
                "\t" + ", ".join([f"{key}={value:.2f}" for key, value in metrics.items()]))

    # writer.add_hparams(hparams_dict, metrics_dict)

    # log hyper parameter
    logger.info("Hyper Parameters")
    for key, value in hparams_dict.items():
        logger.info(f"\t{key}: {value}")

    # log continual learning metrics
    logger.debug("Continual Learning Performance:")
    for key, value in metrics.items():
        logger.info(f"\t{key}: {value}")

    logger.info(f"Task learned: {dataset_getter.tasks}")
    logger.success("Finished Training")


if __name__ == "__main__":
    continual_learning()
