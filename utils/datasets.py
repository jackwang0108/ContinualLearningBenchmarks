# Standard Library
from pathlib import Path
from itertools import chain
from functools import partial
from typing import Literal, Callable, Optional

# Third-Party Library
import numpy as np

# Torch Library
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# My Library
from .data import cifar100 as cifar100
from .transforms import UnifiedTransforms
from .annotation import Images, Labels, Task, Split, SupportedDataset, TorchDatasetImplementation, TaskDataGetter, ClassDataGetter


def get_cls_names(dataset: SupportedDataset) -> list[str]:
    """
    get the name of all classes in the dataset

    Args:
        dataset (SupportedDataset): dataset to load, check `annotation.py` for supported dataset

    Raises:
        NotImplementedError: if the dataset is not supported

    Returns:
        list[str]: name of all classes in the dataset
    """
    if dataset == "cifar100":
        return cifar100.get_cifar100_cls_names()
    else:
        raise NotImplementedError


def get_cls_data_getter(dataset: SupportedDataset, split: Split) -> ClassDataGetter:
    """
    get class_data_getter of dataset

    Args:
        dataset (SupportedDataset): dataset to load, check `annotation.py` for supported dataset
        split (Split): split of the dataset

    Raises:
        NotImplementedError: if the dataset is not supported

    Returns:
        ClassDataGetter: class_data_getter of dataset
    """
    if dataset == "cifar100":
        return cifar100.get_cifar100_cls_data_getter(split)
    else:
        raise NotImplementedError


def get_tasks(dataset: SupportedDataset, cls_names: list[str], task_num: int, fixed_tasks: bool) -> list[Task]:
    """
    split the classes of dataset into tasks


    Args:
        dataset (SupportedDataset): dataset to load, check `annotation.py` for supported dataset
        cls_names (list[str]): name of all classes in the dataset
        task_num (int): number of tasks to generate
        fixed_tasks (bool): if use pre-defined tasks of the dataset.

    Raises:
        NotImplementedError: if the dataset is not supported

    Returns:
        list[Task]: tasks for continual learning
    """
    if dataset == "cifar100":
        return cifar100.get_cifar100_tasks(cls_names, task_num, fixed_tasks)
    else:
        raise NotImplementedError


def get_task_data_getter(dataset: SupportedDataset, split: Split) -> TaskDataGetter:
    """
    returns the task_data_getter of dataset


    Args:
        dataset (SupportedDataset): dataset to load, check `annotation.py` for supported dataset
        split (Split): split of the dataset

    Raises:
        NotImplementedError: if the dataset is not supported

    Returns:
        TaskDataGetter: task_data_getter of the dataset
    """
    if dataset == "cifar100":
        # return cifar100.get_task_data_getter(split)
        return partial(cifar100.get_cifar100_task_data, split)
    else:
        raise NotImplementedError


def get_dataset(dataset: SupportedDataset) -> TorchDatasetImplementation:
    """
    get the torch.utils.data.Dataset implementation of dataset

    Args:
        dataset (SupportedDataset): dataset to load, check `annotation.py` for supported dataset

    Raises:
        NotImplementedError: if the dataset is not supported

    Returns:
        TorchDatasetImplementation: torch.utils.data.Dataset implementation of dataset
    """
    if dataset == "cifar100":
        return cifar100.Cifar100Dataset
    else:
        raise NotImplementedError


class CLDatasetGetter():
    def __init__(
        self,
        dataset: SupportedDataset,
        task_num: int = 10,
        fixed_task: bool = False,
        given_tasks: Optional[list[Task]] = None,
        transform: Optional[nn.Module] = None
    ) -> None:

        self.dataset = dataset
        self.task_num = task_num
        self.fixed_task = fixed_task

        cls_names = get_cls_names(dataset)

        # get tasks
        if fixed_task:
            # if use fixed task defined in data.[dataset].get_[dataset]_tasks
            self.tasks = get_tasks(dataset, cls_names, task_num, fixed_task)
        else:
            # if use specified task or random generated task
            self.tasks = get_tasks(
                dataset, cls_names, task_num, fixed_task) if given_tasks is None else given_tasks
        self.num_cls_per_task = len(self.tasks[0])

        # get cls_id_mapper
        self.cls_id_mapper = {cls_name: cls_id for cls_id,
                              cls_name in enumerate(chain.from_iterable(self.tasks))}

        # get task_data_getter
        self.test_task_data_getter = get_task_data_getter(dataset, "test")
        self.train_task_data_getter = get_task_data_getter(dataset, "train")

        # get transforms
        self.train_transform = UnifiedTransforms(
            is_eval=False, use_crop_transform=False, same_crop_transform=False) if transform is not None else None
        self.test_transform = UnifiedTransforms(
            is_eval=True, use_crop_transform=False, same_crop_transform=False) if transform is not None else None

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
        current_task_cls_ids = [self.cls_id_mapper[cls_name]
                                for cls_name in current_task]

        task_images_train, task_labels_train = self.train_task_data_getter(
            current_task, current_task_cls_ids)

        task_images_test, task_labels_test = self.test_task_data_getter(
            current_task, current_task_cls_ids)

        # get torch.utils.data.Dataset
        dataset = get_dataset(self.dataset)

        train_dataset = dataset(
            task_images_train, task_labels_train, self.train_transform)
        test_dataset = dataset(
            task_images_test, task_labels_test, self.test_transform)

        self.task_id += 1

        return self.task_id - 1, current_task, train_dataset, test_dataset


if __name__ == "__main__":
    from .helper import draw_image

    dataset_getter = CLDatasetGetter("cifar100", 10, False, transform=None)

    for task_id, cls_names, train_dataset, test_dataset in dataset_getter:
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        print(f"{task_id=}, {cls_names=}")
        for image, label in train_loader:
            print(image.shape)
            print(label)
            break

        draw_image(image, "./train-batch.png")

        for image, label in test_loader:
            print(image.shape)
            print(label)
            break

        draw_image(image, "./test-batch.png")

        break
