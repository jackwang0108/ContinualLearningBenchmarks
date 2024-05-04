# Standard Library
from pathlib import Path
from typing import Literal, Callable, Optional

# Third-Party Library
import numpy as np

# Torch Library
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# My Library
from .data import cifar100 as cifar100
from .transforms import UnifiedTransforms


def get_cls_name(dataset: Literal["cifar100"]) -> list[str]:
    if dataset == "cifar100":
        return cifar100.get_cifar100_cls_names()
    else:
        raise NotImplementedError


def get_tasks(dataset: Literal["cifar100"], cls_names: list[str], task_num: int, fixed_tasks: bool) -> list[list[str] | tuple[list[str], list[str]]]:
    if dataset == "cifar100":
        return cifar100.get_cifar100_tasks(cls_names, task_num, fixed_tasks)
    else:
        raise NotImplementedError


def get_task_data_getter(dataset: Literal["cifar100"], split: Literal["train", "val", "test"]) -> Callable[[list[str]], tuple[np.ndarray, list[int]]]:
    if dataset == "cifar100":
        return cifar100.get_task_data_getter(split)


def get_dataset(dataset: Literal["cifar100"]) -> type[cifar100.Cifar100Dataset]:
    if dataset == "cifar100":
        return cifar100.Cifar100Dataset
    else:
        raise NotImplementedError


class CLDatasetGetter():
    def __init__(
        self,
        dataset: Literal["cifar100"],
        task_num: int = 10,
        fixed_task: bool = False,
        given_tasks: Optional[list[str]] = None,
        transform: Optional[nn.Module | bool] = None
    ) -> None:

        self.dataset = dataset
        self.task_num = task_num
        self.fixed_task = fixed_task

        cls_names = get_cls_name(dataset)

        if fixed_task:
            self.tasks = get_tasks(dataset, cls_names, task_num, fixed_task)
        else:
            self.tasks = get_tasks(
                dataset, cls_names, task_num, fixed_task) if given_tasks is None else given_tasks
        self.num_cls_per_task = len(self.tasks[0])

        reordered_cls_names = [
            cls_name for task in self.tasks for cls_name in task]
        self.cls_id_mapper = dict(enumerate(reordered_cls_names))

        # For datasets with label given, the task may shuffle the classes and re-assign a new class id
        self.id_mapper = None
        if dataset in ["cifar100"]:
            origin_cls_name_mapper = {
                cls_name: cls_id for cls_id, cls_name in enumerate(cls_names)}
            reordered_cls_name_mapper = {
                cls_name: cls_id for cls_id, cls_name in enumerate(reordered_cls_names)}
            self.id_mapper = {origin_cls_name_mapper[cls_name]:
                              reordered_cls_name_mapper[cls_name]for cls_name in cls_names}

        # self.val_data_getter = get_task_data_getter(dataset, "val")
        self.test_data_getter = get_task_data_getter(dataset, "test")
        self.train_data_getter = get_task_data_getter(dataset, "train")

        if transform or transform is None:
            self.train_transform = UnifiedTransforms(
                is_eval=False, use_crop_transform=False, same_crop_transform=False)
            self.test_transform = UnifiedTransforms(
                is_eval=True, use_crop_transform=False, same_crop_transform=False)

        self.learned_tasks = []

        self.task_id = 0

    def __iter__(self):
        return self

    def __next__(self) -> tuple[int, list[str], Dataset, Dataset]:
        if self.task_id >= len(self.tasks):
            raise StopIteration

        current_task = self.tasks[self.task_id]

        self.learned_tasks.append(current_task)

        # get train and test data
        task_images_train, task_labels_train = self.train_data_getter(
            current_task, self.id_mapper)

        task_images_test, task_labels_test = self.test_data_getter(
            current_task, self.id_mapper)

        # get dataset
        dataset = get_dataset(self.dataset)
        train_dataset = dataset(
            task_images_train, task_labels_train, self.train_transform)
        test_dataset = dataset(
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
