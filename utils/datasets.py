# Standard Library
import copy
import random
import importlib
from itertools import chain
from functools import partial
from typing import cast, get_args
from collections.abc import Callable
from typing import Optional, Protocol, Literal

# Third-Party Library
import numpy as np

# Torch Library
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# My Library
from .data import conf
from .data import cifar100 as cifar100
from .annotation import (
    Images,
    Labels,
    Task,
    CollateFunc,
    ClassDataGetter,
)


Split = Literal["train", "val", "test"]
AvailableDatasets = Literal["cifar100"]
available_datasets = get_args(AvailableDatasets)


class CLDatasetModule(Protocol):

    def get_cls_names(self) -> list[str]:
        """
        get the name of all classes in the dataset

        Args:
            dataset (SupportedDataset): dataset to load, check `annotation.py` for supported dataset

        Raises:
            NotImplementedError: if the dataset is not supported

        Returns:
            list[str]: name of all classes in the dataset
        """
        pass

    def get_transforms(self) -> tuple[transforms.Compose, transforms.Compose]:
        pass

    def get_cls_data_getter(self, split: Split) -> ClassDataGetter:
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
        pass

    def get_task_data(self, split: Split) -> tuple[Images, Labels]:
        """
        get all data of the given task

        Args:
            split (Split): split of the dataset

        Returns:
            tuple[Images, Labels]: the images and labels of give task
        """
        pass

    def collate_fn(
        self, batched_data: list[tuple[torch.FloatTensor, np.ndarray, np.int64]]
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        pass


def get_tasks(
    dataset: AvailableDatasets, cls_names: list[str], task_num: int, fixed_tasks: bool
) -> list[Task]:
    if fixed_tasks:
        return conf.fixed_task[dataset]

    # Note: shuffle is applied directly on the cls_names passed in
    random.shuffle(cls_names)
    cls_num = len(cls_names) // task_num
    return [cls_names[i : i + cls_num] for i in range(0, len(cls_names), cls_num)]


def default_collate_fn(
    batched_data: list[tuple[torch.FloatTensor, np.ndarray, np.int64]]
) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:

    augmented_images, original_images, labels = zip(*batched_data)

    augmented_images = torch.stack(augmented_images, dim=0).float()

    original_images = (
        torch.from_numpy(np.stack(original_images, axis=0)).float().permute(0, 3, 1, 2)
    )

    labels = torch.from_numpy(np.array(labels)).long()

    return augmented_images, original_images, labels


class CLDatasetGetter:
    def __init__(
        self,
        dataset: AvailableDatasets,
        task_num: int = 10,
        fixed_task: bool = False,
        given_tasks: Optional[list[Task]] = None,
    ) -> None:
        # sourcery skip: assign-if-exp

        assert (
            dataset in available_datasets
        ), f"unsupported dataset: {dataset}, current available datasets: {available_datasets}"

        self.dataset = dataset
        self.task_num = task_num
        self.fixed_task = fixed_task

        # load the dataset module
        self.dataset_module = cast(
            CLDatasetModule, importlib.import_module(f"utils.data.{dataset}")
        )

        # get class names and tasks
        if given_tasks or fixed_task:
            # if user has given a task list or using the predefined task list
            self.tasks = given_tasks or get_tasks(dataset, None, None, True)
            cls_names = list(chain.from_iterable(self.tasks))
        else:
            # else using a random generated task list
            cls_names = self.dataset_module.get_cls_names()
            self.tasks = get_tasks(dataset, cls_names, task_num, False)

        self.num_cls_per_task = len(self.tasks[0])

        # get cls_id_mapper
        self.cls_id_mapper = {
            cls_name: cls_id for cls_id, cls_name in enumerate(cls_names)
        }

        # get task_data_getter
        self.test_task_data_getter = partial(self.dataset_module.get_task_data, "test")
        self.train_task_data_getter = partial(
            self.dataset_module.get_task_data, "train"
        )

        # get training and testing transforms
        self.train_transform, self.test_transform, self.denorm_transform = (
            self.dataset_module.get_transforms()
        )

        self.learned_tasks: list[Task] = []

        self.current_task: Task = None
        self.current_task_id: int = -1

        self.test_datasets: list[Dataset] = []
        self.train_datasets: list[Dataset] = []

    def get_collate_fn(self, allow_default: bool = False) -> CollateFunc:
        collate_fn = getattr(self.dataset_module, "collate_fn", None)
        if collate_fn is None and allow_default:
            collate_fn = default_collate_fn
        assert (
            collate_fn is not None
        ), f"no collate_fn provided for dataset {self.dataset}"
        return collate_fn

    def __iter__(self):
        return self

    def __next__(self) -> tuple[int, Task, Dataset, Dataset]:

        self.current_task_id += 1
        if self.current_task_id >= len(self.tasks):
            raise StopIteration

        self.current_task = self.tasks[self.current_task_id]

        self.learned_tasks.append(self.current_task)

        # get train and test data
        current_task_cls_ids = [
            self.cls_id_mapper[cls_name] for cls_name in self.current_task
        ]

        task_images_train, task_labels_train = self.train_task_data_getter(
            self.current_task, current_task_cls_ids
        )

        task_images_test, task_labels_test = self.test_task_data_getter(
            self.current_task, current_task_cls_ids
        )

        # get torch.utils.data.Dataset
        dataset: cifar100.Cifar100Dataset = getattr(
            self.dataset_module, f"{self.dataset.capitalize()}Dataset"
        )

        train_dataset = dataset(
            task_images_train,
            task_labels_train,
            self.train_transform,
            self.denorm_transform,
        )
        test_dataset = dataset(
            task_images_test,
            task_labels_test,
            self.test_transform,
            self.denorm_transform,
        )

        test_dataset.collate_fn = self.get_collate_fn(allow_default=True)
        train_dataset.collate_fn = self.get_collate_fn(allow_default=True)

        return self.current_task_id, self.current_task, train_dataset, test_dataset


if __name__ == "__main__":
    import torch
    from typing import get_args
    from .helper import draw_image

    def test_CLDatasetGetter(dataset: AvailableDatasets, task_num: int):
        assert dataset in available_datasets, f"Invalid {dataset=}"

        dataset_getter = CLDatasetGetter("cifar100", task_num, False)

        train_dataset: cifar100.Cifar100Dataset
        test_dataset: cifar100.Cifar100Dataset
        # sourcery skip: no-loop-in-tests
        for task_id, cls_names, train_dataset, test_dataset in dataset_getter:
            train_loader = DataLoader(
                train_dataset,
                batch_size=32,
                shuffle=True,
                collate_fn=default_collate_fn,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=32,
                shuffle=False,
                collate_fn=default_collate_fn,
            )

            draw_image(
                torch.from_numpy(
                    train_dataset.images[: 500 * (100 // task_num) : 500]
                ).permute(0, 3, 1, 2),
                f"./test/{test_CLDatasetGetter.__name__}-{dataset=}-{task_num=}-{task_id=}.png",
            )

            print(f"{task_id=}, {cls_names=}")
            for augmented_image, original_image, label in train_loader:
                print(augmented_image.shape)
                print(original_image.shape)
                print(label)
                break

            draw_image(
                augmented_image,
                f"./test/{test_CLDatasetGetter.__name__}-{dataset=}-{task_num=}-{task_id=}-train-augmented.png",
            )

            draw_image(
                original_image.int(),
                f"./test/{test_CLDatasetGetter.__name__}-{dataset=}-{task_num=}-{task_id=}-train-original.png",
            )

            for augmented_image, original_image, label in test_loader:
                print(augmented_image.shape)
                print(original_image.shape)
                print(label)
                break

            draw_image(
                augmented_image,
                f"./test/{test_CLDatasetGetter.__name__}-{dataset=}-{task_num=}-{task_id=}-test-augmented.png",
            )

            draw_image(
                original_image.int(),
                f"./test/{test_CLDatasetGetter.__name__}-{dataset=}-{task_num=}-{task_id=}-test-original.png",
            )

    test_CLDatasetGetter("cifar100", task_num=2)
    test_CLDatasetGetter("cifar100", task_num=5)
    test_CLDatasetGetter("cifar100", task_num=10)
