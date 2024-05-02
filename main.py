# Standard Library

# Third-Party Library

# Torch Library
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# My Library
from utils.loader import CLDatasetGetter


def continual_learning():

    dataset_getter = CLDatasetGetter(
        dataset="cifar100", task_num=10, fixed_task=False)

    for task_id, current_task, test_classes, train_dataset, val_dataset, test_dataset in dataset_getter:
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        image: torch.FloatTensor
        label: torch.FloatTensor

        # learn the task
        print(f"{task_id=}, {current_task=}, {test_classes=}")
        for image, label in train_loader:
            print(image.shape)
            print(label.shape)
            break

        # validate the task
        for image, label in val_loader:
            print(image.shape)
            print(label.shape)
            break

        # test the task
        for image, label in test_loader:
            print(image.shape)
            print(label.shape)
            break


if __name__ == "__main__":
    continual_learning()
