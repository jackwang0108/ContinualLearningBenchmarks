# Standard Library
import datetime
from pathlib import Path
from typing import Callable, Optional

# Third-Party Library
import numpy as np

# Torch Library
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# My Library
from model.finetune import Finetune
from utils.helper import get_logger
from utils.datasets import CLDatasetGetter
from utils.helper import (plot_matrix, draw_image)
from utils.helper import (get_probas, get_pred, to_onehot)
from utils.helper import (get_top1_acc,
                          get_backward_transfer,
                          get_last_setp_accuracy,
                          get_average_incremental_accuracy,
                          get_forgetting_rate,
                          CLMetrics)
from utils.annotation import (
    TaskLearner,
    PerformanceFunc,
    CLAbilityTester,
)

rname = input("the name of this running: ")
wname = f"{rname}-{datetime.datetime.now().strftime('%m-%d %H.%M')}"
writer = SummaryWriter(log_dir := f"log/{wname}")
logger = get_logger(Path(log_dir) / "running.log")
device = torch.device("cuda:0")

hparams_dict = {}


def train_epoch(model: Finetune, train_loader: DataLoader, loss_func: nn.Module, optimizer: optim.Optimizer) -> torch.FloatTensor:
    total_loss = 0

    total_num_classes = len(model.learned_classes) + len(model.current_task)

    loss: torch.FloatTensor
    image: torch.FloatTensor
    label: torch.LongTensor
    for image, label in train_loader:
        image = image.to(device)
        label = to_onehot(label, total_num_classes).to(device)

        logits = model(image)
        loss = loss_func(logits, label)

        total_loss += loss.clone().detach()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(train_loader)


@torch.no_grad()
def test_epoch(model: Finetune, test_loader: DataLoader, perf_func: PerformanceFunc, num_cls_per_task: int) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    performance = []

    for image, label in test_loader:
        image, label = image.to(device), label.to(device)
        logits = model(image)
        probas = get_probas(logits)
        preds = get_pred(probas)
        performance.append(
            perf_func(preds, label, len(model.learned_classes) + num_cls_per_task))

    return sum(performance) / len(performance)


def get_task_learner() -> TaskLearner:
    num_task_learned = 0

    def task_learner(task_id: int, num_cls_per_task: int, model: Finetune, train_loader: DataLoader, test_loader: DataLoader) -> Finetune:

        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

        log_times = 5
        num_epoch = 70
        for epoch in range(num_epoch):
            train_loss = train_epoch(model, train_loader, loss_func, optimizer)
            test_top1_acc = test_epoch(model, test_loader, get_top1_acc,
                                       num_cls_per_task)

            if (epoch + 1) % (num_epoch // log_times) == 0:
                logger.info(
                    f"\tEpoch [{num_epoch}/{epoch+1:>{len(str(num_epoch))}d}], {train_loss=:.2f}, {test_top1_acc=:.2f}")

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

    @torch.no_grad
    def continual_learning_ability_tester(task_id: int, learned_tasks: list[list[str]], learned_task_loaders: list[DataLoader], model: Finetune) -> tuple[np.ndarray, Optional[dict[str, float]]]:
        nonlocal cl_matrix, num_cls_per_task

        # test on all tasks, including previous and current task
        for i, previous_loader in enumerate(learned_task_loaders):

            # using past classifier
            # model.current_classifier = model.classifiers[i]
            # cl_matrix[i, task_id] = test_epoch(
            #     model, previous_loader, get_top1_acc, num_cls_per_task)

            logger.info(f"\ttest on task {i}, test_acc={
                cl_matrix[i, task_id]:.2f}, {learned_tasks[i]}")

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

        return cl_matrix, current_metrics if task_id >= 1 else None

    return continual_learning_ability_tester


def continual_learning():

    dataset_getter = CLDatasetGetter(
        dataset="cifar100", task_num=10, fixed_task=True)

    model = Finetune().to(device)

    # logging
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info("Task List:")
    for task_id, task in enumerate(dataset_getter.tasks):
        logger.info(f"\tTask {task_id}: {task}")

    # get task learner and cl-ability tester
    task_learner = get_task_learner()

    continual_learning_ability_tester = get_continual_learning_ability_tester(
        dataset_getter.task_num, dataset_getter.num_cls_per_task)

    learned_tasks: list[DataLoader] = []
    learned_task_loaders: list[DataLoader] = []
    for task_id, current_task, train_dataset, test_dataset in dataset_getter:
        # prepare the data for the task
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

        # learn the new task
        with model.set_new_task(current_task):

            logger.success(f"{task_id=}, {current_task=}")
            model = task_learner(
                task_id, dataset_getter.num_cls_per_task, model, train_loader, test_loader)

        # save the test loader for continual learning testing
        learned_tasks.append(current_task)
        learned_task_loaders.append(test_loader)

        # test continual learning performance
        cl_matrix, metrics = continual_learning_ability_tester(
            task_id, learned_tasks, learned_task_loaders, model)

        if metrics is not None:

            logger.debug(
                "\t " + ", ".join([f"{key}={value:.2f}" for key, value in metrics.items()]))

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
