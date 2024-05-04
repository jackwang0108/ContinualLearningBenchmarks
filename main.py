# Standard Library
import datetime
from typing import Callable

# Third-Party Library
import numpy as np

# Torch Library
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# My Library
from model import Model
from model.LingoCL import LingoCL
from model.finetune import Finetune
from utils.loader import CLDatasetGetter
from utils.helper import get_probas, get_pred
from utils.helper import plot_matrix, draw_image
from utils.helper import (get_top1_acc,
                          get_backward_transfer,
                          get_last_setp_accuracy,
                          get_average_incremental_accuracy,
                          get_forgetting_rate)

rname = input("the name of this running: ")
wname = f"{rname}-{datetime.datetime.now().strftime('%m-%d %H.%M')}"
writer = SummaryWriter(f"log/{wname}")
device = torch.device("cuda:0")

hparams_dict = {}
metrics_dict = {}


def train_epoch(model: Model, train_loader: DataLoader, loss_func: nn.Module, optimizer: optim.Optimizer) -> torch.FloatTensor:
    total_loss = 0

    loss: torch.FloatTensor
    image: torch.FloatTensor
    label: torch.LongTensor
    for image, label in train_loader:
        image, label = image.to(device), label.to(device)

        logits = model(image)
        loss = loss_func(logits, label)

        total_loss += loss.clone().detach()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(train_loader)


@torch.no_grad()
def test_epoch(model: Model, test_loader: DataLoader, perf_func: Callable[[torch.FloatTensor, torch.FloatTensor, int], torch.FloatTensor], num_cls_per_task: int) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    performance = []

    for image, label in test_loader:
        image, label = image.to(device), label.to(device)
        logits = model(image)
        probas = get_probas(logits)
        preds = get_pred(probas)
        performance.append(
            perf_func(preds, label, len(model.learned_classes) + num_cls_per_task))

    return sum(performance) / len(performance)


def get_task_learner() -> Callable[[int, int, Model, DataLoader, DataLoader], Model]:
    num_task_learned = 0

    def task_learner(task_id: int, num_cls_per_task: int, model: Model, train_loader: DataLoader, test_loader: DataLoader) -> Model:

        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

        print_time = 5
        num_epoch = 5
        for epoch in range(num_epoch):
            train_loss = train_epoch(model, train_loader, loss_func, optimizer)
            test_acc = test_epoch(model, test_loader, get_top1_acc,
                                  num_cls_per_task)

            if epoch % (num_epoch // print_time) == 0:
                print(
                    f"\tEpoch [{num_epoch}/{epoch+1:>{len(str(num_epoch))}d}], {train_loss=:.2f}, {test_acc=:.2f}")

            writer.add_scalar("task-learning/train-loss",
                              scalar_value=train_loss, global_step=epoch + task_id * num_epoch)
            writer.add_scalar(
                "task-learning/test-top1-acc", scalar_value=test_acc, global_step=epoch + task_id * num_epoch)

        # log hparams
        nonlocal num_task_learned
        if num_task_learned == 0:
            hparams_dict["num_epoch"] = num_epoch
            hparams_dict["optim"] = optimizer.__class__.__name__
            hparams_dict["lr"] = optimizer.defaults["lr"]
            hparams_dict["momentum"] = optimizer.defaults["momentum"]
            hparams_dict["weight_decay"] = optimizer.defaults["weight_decay"]
            hparams_dict["dampening"] = optimizer.defaults["dampening"]
        num_task_learned += 1

        return model
    return task_learner


def get_continual_learning_ability_tester(task_num: int, num_cls_per_task: int) -> Callable[[int, list[list[str]], list[DataLoader], Model], np.ndarray]:

    num_cls_per_task = num_cls_per_task
    cl_matrix = np.zeros((task_num, task_num))

    @torch.no_grad
    def continual_learning_ability_tester(task_id: int, learned_tasks: list[list[str]], learned_task_loaders: list[DataLoader], model: Model) -> np.ndarray:
        nonlocal cl_matrix, num_cls_per_task

        # test on all tasks, including previous and current task
        for i, previous_loader in enumerate(learned_task_loaders):
            cl_matrix[i, task_id] = test_epoch(
                model, previous_loader, get_top1_acc, num_cls_per_task)

            print(f"\ttest on task {i}, test_acc={
                  cl_matrix[i, task_id]:.2f}, {learned_tasks[i]}")

        # calculate continual learning ability metrics and log to summarywriter
        if task_id >= 1:
            current_cl_matrix = cl_matrix[:task_id+1, :task_id+1]
            writer.add_scalar(
                tag="continual-learning-metrics/backward-transfer",
                scalar_value=(bwt := get_backward_transfer(current_cl_matrix)),
                global_step=task_id,
            )
            writer.add_scalar(
                tag="continual-learning-metrics/forgetting-rate",
                scalar_value=(
                    forget := get_forgetting_rate(current_cl_matrix)),
                global_step=task_id,
            )
            writer.add_scalar(
                tag="continual-learning-metrics/last-step-accuracy",
                scalar_value=(
                    last := get_last_setp_accuracy(current_cl_matrix)),
                global_step=task_id,
            )
            writer.add_scalar(
                tag="continual-learning-metrics/average-incremental-accuracy",
                scalar_value=(avg := get_average_incremental_accuracy(
                    current_cl_matrix
                )),
                global_step=task_id,
            )

        # draw heatmap of cl_matrix and log to summarywriter
        writer.add_figure(
            tag="cl_matrix",
            figure=plot_matrix(cl_matrix, task_id),
            global_step=task_id,
        )

        return cl_matrix, {"hparam/bwt": bwt, "hparam/forgetting-rate": forget, "hparam/last-step-acc": last, "hparam/average-acc": avg} if task_id >= 1 else None

    return continual_learning_ability_tester


def continual_learning():

    dataset_getter = CLDatasetGetter(
        dataset="cifar100", task_num=10, fixed_task=False)

    model = LingoCL().to(device)

    # get task learner and cl-ability tester
    task_learner = get_task_learner()
    task_learner: Callable[[int, int, Model, DataLoader, DataLoader], Model]

    continual_learning_ability_tester: Callable[[
        int, list[list[str]], list[DataLoader], Finetune], np.ndarray]
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

            print(f"{task_id=}, {current_task=}")
            model = task_learner(
                task_id, dataset_getter.num_cls_per_task, model, train_loader, test_loader)

        # save the test loader for continual learning testing
        learned_tasks.append(current_task)
        learned_task_loaders.append(test_loader)

        # test continual learning performance
        cl_matrix, metrics = continual_learning_ability_tester(
            task_id, learned_tasks, learned_task_loaders, model)

        if metrics is not None:
            metrics_dict.update(metrics)

    # writer.add_hparams(hparams_dict, metrics_dict)


if __name__ == "__main__":
    continual_learning()
