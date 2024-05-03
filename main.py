# Standard Library
import datetime
from typing import Callable

# Third-Party Library

# Torch Library
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# My Library
from model.finetune import Finetune
from utils.loader import CLDatasetGetter
from utils.helper import to_onehot, get_probas, get_pred, get_acc

writer = SummaryWriter(
    f"log/{datetime.datetime.now().strftime('%m-%d %H.%M')}")
device = torch.device("cuda:0")


def train_epoch(model: Finetune, train_loader: DataLoader, loss_func: nn.Module, optimizer: optim.Optimizer) -> torch.FloatTensor:
    total_loss = 0

    loss: torch.FloatTensor
    image: torch.FloatTensor
    label: torch.IntTensor
    for image, label in train_loader:
        image, label = image.to(device), label.to(device)

        logits = model(image)
        loss = loss_func(logits, label)

        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(train_loader)


@torch.no_grad()
def val_epoch(model: Finetune, val_loader: DataLoader, perf_func: Callable[[torch.FloatTensor, torch.FloatTensor, int], torch.FloatTensor], num_cls_per_task: int) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    performance = []

    for image, label in val_loader:
        image, label = image.to(device), label.to(device)
        logits = model(image)
        probas = get_probas(logits)
        preds = get_pred(probas)
        performance.append(
            perf_func(preds, label, len(model.learned_classes) + num_cls_per_task))

    return sum(performance) / len(performance)


def learn_task(task_id: int, num_cls_per_task: int, model: Finetune, train_loader: DataLoader, val_loader: DataLoader) -> nn.Module:

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

    num_epoch = 50
    for epoch in range(num_epoch):
        train_loss = train_epoch(model, train_loader, loss_func, optimizer)
        val_acc = val_epoch(model, val_loader, get_acc,
                            num_cls_per_task)

        writer.add_scalar("train/train-loss",
                          scalar_value=train_loss, global_step=epoch + task_id * num_epoch)
        writer.add_scalar(
            "train/val-acc", scalar_value=val_acc, global_step=epoch + task_id * num_epoch)

    return model


def test_continual_learning_ability():
    pass


def continual_learning():

    dataset_getter = CLDatasetGetter(
        dataset="cifar100", task_num=20, fixed_task=False)

    model = Finetune().to(device)

    learned_task: list[list[str]] = []
    for task_id, current_task, test_classes, train_dataset, val_dataset, test_dataset in dataset_getter:
        # prepare the data for the task
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

        # learn the new task
        with model.set_new_task(current_task):

            print(f"{task_id=}, {current_task=}")
            model = learn_task(
                task_id, dataset_getter.num_cls_per_task, model, train_loader, val_loader)

        # test continual learning performance
        if task_id >= 1:
            test_continual_learning_ability()

        learned_task.append(current_task)


if __name__ == "__main__":
    continual_learning()
