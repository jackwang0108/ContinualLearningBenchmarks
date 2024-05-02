# Standard Library
from pathlib import Path
from typing import Literal, Callable

# Third-Party Library
import numpy as np

# Torch Library
from torch.utils.data import DataLoader

# My Library
import data.cifar100 as cifar100
from transforms import UnifiedTransforms


def get_cls_name(dataset: Literal["cifar100"]) -> list[str]:
    if dataset == "cifar100":
        return cifar100.get_cifar100_cls_names()
    else:
        raise NotImplementedError


def get_tasks(dataset: Literal["cifar100"], cls_names: list[str], task_num: int, fixed_tasks: bool) -> list[list[str]]:
    if dataset == "cifar100":
        return cifar100.get_cifar100_tasks(cls_names, task_num, fixed_tasks)
    else:
        raise NotImplementedError


def get_task_data_getter(dataset: Literal["cifar100"], split: Literal["train", "val", "test"]) -> Callable[[list[str]], tuple[np.ndarray, list[int]]]:
    if dataset == "cifar100":
        return cifar100.get_task_data_getter(split)


class CLDatasetGetter():
    def __init__(
        self,
        dataset: Literal["cifar100"],
        task_num: int = 10,
        fixed_task: bool = False
    ) -> None:

        self.task_num = task_num
        self.fixed_task = fixed_task

        cls_name = get_cls_name(dataset)

        self.tasks = get_tasks(dataset, cls_name, task_num, fixed_task)

        self.test_data_getter = get_task_data_getter(dataset, "test")
        self.train_data_getter = get_task_data_getter(dataset, "train")

        self.train_transform = UnifiedTransforms(
            is_eval=False, use_crop_transform=False, same_crop_transform=False)
        self.test_transform = UnifiedTransforms(
            is_eval=True, use_crop_transform=False, same_crop_transform=False)

        self.task_id = 0

    def __iter__(self):
        return self

    def __next__(self) -> tuple[int, list[str], cifar100.Cifar100Dataset, cifar100.Cifar100Dataset]:
        if self.task_id >= len(self.tasks):
            raise StopIteration

        current_task = self.tasks[self.task_id]
        task_images_train, task_labels_train = self.train_data_getter(
            current_task)

        task_images_test, task_labels_test = self.test_data_getter(
            current_task)

        train_dataset = cifar100.Cifar100Dataset(
            task_images_train, task_labels_train, self.train_transform)
        test_dataset = cifar100.Cifar100Dataset(
            task_images_test, task_labels_test, self.test_transform)

        self.task_id += 1
        return self.task_id - 1, current_task, train_dataset, test_dataset


if __name__ == "__main__":
    dataset_getter = CLDatasetGetter("cifar100", 10, False)

    for task_id, cls_names, train_dataset, test_dataset in dataset_getter:
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        print(f"{task_id=}, {cls_names=}")
        for image, label in train_loader:
            print(image.shape)
            print(label.shape)
            break

        for image, label in test_loader:
            print(image.shape)
            print(label.shape)
            break
