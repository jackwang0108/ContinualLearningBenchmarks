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
from model.finetune import Finetune
from utils.loader import CLDatasetGetter
from utils.helper import get_probas, get_pred, get_top1_acc, plot_matrix, draw_image

wname = datetime.datetime.now().strftime('%m-%d %H.%M')
writer = SummaryWriter(f"log/{wname}")
device = torch.device("cuda:0")


def train_epoch(model: Finetune, train_loader: DataLoader, loss_func: nn.Module, optimizer: optim.Optimizer) -> torch.FloatTensor:
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
def test_epoch(model: Finetune, test_loader: DataLoader, perf_func: Callable[[torch.FloatTensor, torch.FloatTensor, int], torch.FloatTensor], num_cls_per_task: int) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    performance = []

    for image, label in test_loader:
        image, label = image.to(device), label.to(device)
        logits = model(image)
        probas = get_probas(logits)
        preds = get_pred(probas)
        performance.append(
            perf_func(preds, label, len(model.learned_classes) + num_cls_per_task))

    return sum(performance) / len(performance)


def learn_task(task_id: int, num_cls_per_task: int, model: Finetune, train_loader: DataLoader, test_loader: DataLoader) -> nn.Module:

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

    print_time = 5
    num_epoch = 50
    for epoch in range(num_epoch):
        train_loss = train_epoch(model, train_loader, loss_func, optimizer)
        test_acc = test_epoch(model, test_loader, get_top1_acc,
                              num_cls_per_task)

        if epoch % (num_epoch // print_time) == 0:
            print(
                f"\tEpoch [{num_epoch}/{epoch:>{len(str(num_epoch))}d}], {train_loss=:.2f}, {test_acc=:.2f}")

        writer.add_scalar("task-learning/train-loss",
                          scalar_value=train_loss, global_step=epoch + task_id * num_epoch)
        writer.add_scalar(
            "task-learning/test-top1-acc", scalar_value=test_acc, global_step=epoch + task_id * num_epoch)

    return model


def get_continual_learning_ability_tester(task_num: int, num_cls_per_task: int) -> Callable[[int, list[list[str]], list[DataLoader], Finetune], np.ndarray]:

    num_cls_per_task = num_cls_per_task
    cl_matrix = np.zeros((task_num, task_num))

    @torch.no_grad
    def continual_learning_ability_tester(task_id: int, learned_tasks: list[list[str]], learned_task_loaders: list[DataLoader], model: Finetune) -> np.ndarray:

        # test current task
        # current_loader = learned_tasks[-1]
        # cl_matrix[task_id, task_id] = test_epoch(
        #     model, current_loader, get_top1_acc, num_cls_per_task)

        # test previous tasks
        for i, previous_loader in enumerate(learned_task_loaders):
            cl_matrix[i, task_id] = test_epoch(
                model, previous_loader, get_top1_acc, num_cls_per_task)

            print(f"\ttest on task {i}, {learned_tasks[i]}, test_acc={
                  cl_matrix[i, task_id]:.2f}")

        writer.add_figure(tag=f"{wname}", figure=plot_matrix(
            cl_matrix), global_step=task_id)

    return continual_learning_ability_tester


def continual_learning():

    dataset_getter = CLDatasetGetter(
        dataset="cifar100", task_num=20, fixed_task=False)

    model = Finetune().to(device)

    learned_tasks: list[DataLoader] = []
    learned_task_loaders: list[DataLoader] = []
    continual_learning_ability_tester: Callable[[int, list[list[str]], list[DataLoader], Finetune], np.ndarray] = get_continual_learning_ability_tester(
        dataset_getter.task_num, dataset_getter.num_cls_per_task)
    for task_id, current_task, train_dataset, test_dataset in dataset_getter:
        # prepare the data for the task
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

        # learn the new task
        with model.set_new_task(current_task):

            print(f"{task_id=}, {current_task=}")
            model = learn_task(
                task_id, dataset_getter.num_cls_per_task, model, train_loader, test_loader)

        # save the test loader for continual learning testing
        learned_tasks.append(current_task)
        learned_task_loaders.append(test_loader)

        # test continual learning performance
        # TODO: finish this method
        continual_learning_ability_tester(
            task_id, learned_tasks, learned_task_loaders, model)


if __name__ == "__main__":
    continual_learning()
