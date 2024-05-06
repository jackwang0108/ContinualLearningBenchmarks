# Standard Library
import copy
import random
import pickle
from pathlib import Path
from typing import Optional
from functools import lru_cache

# Third-Party Library
import numpy as np

# Torch Library
import torch
import torch.nn as nn
import torchvision.transforms
from torch.utils.data import Dataset

# My Library
from ..annotation import Task, Images, Labels, Split, ClassDataGetter


CIFAR_PATH: Path = (Path(__file__).resolve() /
                    "../../../data/cifar-100-python").resolve()


@lru_cache(maxsize=1)
def check_cifar100(assertion: bool = True) -> bool:
    if assertion:
        assert CIFAR_PATH.exists(), f"Cifar100 is not downloaded! Please downloaded Cifar100 and extract to {
            CIFAR_PATH.relative_to(Path(__file__).parent.parent.parent.resolve())}"
    return CIFAR_PATH.exists()


@lru_cache(maxsize=1)
def get_cifar100_cls_names() -> list[str]:

    check_cifar100(assertion=True)

    meta = CIFAR_PATH / "meta"
    with meta.open(mode="rb") as f:
        data = pickle.load(f)

    return data['fine_label_names']


def get_cifar100_tasks(cls_names: list[str], task_num: int = 10, fixed_tasks: bool = False) -> list[Task]:
    if fixed_tasks:
        return [
            ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver',
                'bed', 'bee', 'beetle', 'bicycle', 'bottle'],
            ['bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
                'can', 'castle', 'caterpillar', 'cattle'],
            ['chair', 'chimpanzee', 'clock', 'cloud', 'cockroach',
                'couch', 'crab', 'crocodile', 'cup', 'dinosaur'],
            ['dolphin', 'elephant', 'flatfish', 'forest', 'fox',
                'girl', 'hamster', 'house', 'kangaroo', 'keyboard'],
            ['lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
                'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain'],
            ['mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter',
                'palm_tree', 'pear', 'pickup_truck', 'pine_tree'],
            ['plain', 'plate', 'poppy', 'porcupine', 'possum',
                'rabbit', 'raccoon', 'ray', 'road', 'rocket'],
            ['rose', 'sea', 'seal', 'shark', 'shrew', 'skunk',
                'skyscraper', 'snail', 'snake', 'spider'],
            ['squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
                'tank', 'telephone', 'television', 'tiger', 'tractor'],
            ['train', 'trout', 'tulip', 'turtle', 'wardrobe',
                'whale', 'willow_tree', 'wolf', 'woman', 'worm']
        ]

    cls_names = copy.deepcopy(cls_names)
    random.shuffle(cls_names)
    cls_num = len(cls_names) // task_num
    return [
        cls_names[i * cls_num: (i + 1) * cls_num]
        for i in range(task_num)
    ]


@lru_cache(maxsize=3)
def get_cifar100_data(split: Split) -> tuple[Images, Labels]:
    # sourcery skip: remove-redundant-if
    file = CIFAR_PATH / ("test" if split == "test" else "train")

    with file.open(mode="rb") as f:
        data = pickle.load(f, encoding="bytes")

    images = data[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.array(data[b"fine_labels"], dtype=np.int64)

    # these codes are not used
    if split != "test" and False:
        cls_id_mapper = {cls_name: cls_id for cls_id,
                         cls_name in enumerate(get_cifar100_cls_names())}

        cls_masks = [labels == cls_id for cls_id in cls_id_mapper.values()]

        # use first 450 as train, last 50 as validation
        start, end = (0, 450) if split == "val" else (449, -1)
        images_list, labels_list = [], []
        for mask in cls_masks:
            true_indices = np.where(mask)[0]
            mask[true_indices[start:end]] = False
            images_list.append(images[mask])
            labels_list.append(labels[mask])
        images, labels = np.concatenate(
            images_list), np.concatenate(labels_list)

    return images, labels


@lru_cache(maxsize=3)
def get_cifar100_cls_data_getter(split: Split) -> ClassDataGetter:

    # get the cls_id
    cls_names = get_cifar100_cls_names()
    cls_id_mapper = {name: id for id, name in enumerate(cls_names)}

    # read data
    images: Images
    labels: Labels
    images, labels = get_cifar100_data(split)

    def cls_data_getter(cls_name: str, new_cls_id: int) -> tuple[Images, Labels]:

        # get class data
        cls_mask = labels == cls_id_mapper[cls_name]
        cls_images = images[cls_mask]
        cls_labels = np.full((cls_images.shape[0],), fill_value=new_cls_id)

        return cls_images, cls_labels

    return cls_data_getter


def get_cifar100_task_data(split: Split, task: Task, cls_ids: list[int]) -> tuple[Images, Labels]:

    images: list[Images]
    labels: list[Labels]
    images, labels = [], []

    cls_data_getter = get_cifar100_cls_data_getter(split)

    for cls_id, cls_name in zip(cls_ids, task):
        cls_image, cls_label = cls_data_getter(cls_name, cls_id)
        images.append(cls_image)
        labels.append(cls_label)

    return np.concatenate(images), np.concatenate(labels)


class Cifar100Dataset(Dataset):
    def __init__(self, images: Images, labels: Labels, transforms: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.images = images
        self.labels = labels
        self.transforms = transforms

        self.to_tensor = torchvision.transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        image = self.to_tensor(self.images[index])

        # data augmentation
        image = self.augment(image)

        return image, self.labels[index]

    def augment(self, image: torch.FloatTensor) -> torch.FloatTensor:
        return image if self.transforms is None else self.transforms(image)

        # @lru_cache(maxsize=3)
        # def get_task_data_getter(split: Split) -> Callable[[Task], tuple[Images, Labels]]:
        #     """
        #     deprecated
        #     """
        #     images, labels = get_cifar100_data(split)

        #     labels = np.array(labels, dtype=np.int64)

        #     cls_id_mapper = {cls_name: cls_id for cls_id,
        #                      cls_name in enumerate(get_cifar100_cls_names())}

        #     def task_data_getter(task: Task, id_mapper: dict[int, int]) -> tuple[Images, Labels]:
        #         task_images, task_labels = [], []

        #         for cls_name in task:
        #             ori_id = cls_id_mapper[cls_name]
        #             cls_mask = labels == ori_id
        #             task_images.append(images[cls_mask])

        #             task_label = labels[cls_mask].copy()

        #             # remap the id
        #             task_label[:] = id_mapper[ori_id]

        #             task_labels.append(task_label)

        #         return np.concatenate(task_images), np.concatenate(task_labels)

        #     return task_data_getter


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from ..helper import draw_image

    cls_names = get_cifar100_cls_names()

    # get the data of a class
    cls_data_getter = get_cifar100_cls_data_getter("train")
    camel_image, camel_label = cls_data_getter("camel", 10)
    print(f"{camel_label=}")
    print(f"{camel_image.shape=}")
    camel_image_tensor = torch.from_numpy(camel_image[:16]).permute(0, 3, 1, 2)
    draw_image(camel_image_tensor, "./get-class-data-example.png")

    # get the data of a task
    task = get_cifar100_tasks(cls_names, 10, False)[0]
    task_image, task_label = get_cifar100_task_data(
        "train", task, range(len(task)))

    print(f"{task=}")
    print(f"{task_label=}")
    print(f"{task_image.shape=}")
    task_image_tensor = torch.from_numpy(task_image[:16]).permute(0, 3, 1, 2)
    draw_image(task_image_tensor,
               "./get-task-data-example.png")
