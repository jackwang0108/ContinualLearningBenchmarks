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
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

# My Library
from ..annotation import Task, Images, Labels, Split, ClassDataGetter


CIFAR_PATH: Path = (Path(__file__).resolve() /
                    "../../../data/cifar-100-python").resolve()

CIFAR_INFO = {
    "url": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
    "filename": "cifar-100-python.tar.gz",
    "tgz_md5": "eb9058c3a382ffc7106e4002c42a8d85",
    "file_list": [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
        ["meta",  "7973b15100ade9c7d40fb424638fde48"]
    ]
}


@lru_cache(maxsize=1)
def check_and_download_cifar100() -> bool:
    if not CIFAR_PATH.exists():
        download_and_extract_archive(
            url=CIFAR_INFO['url'],
            extract_root=CIFAR_PATH.parent,
            download_root=CIFAR_PATH.parent,
            filename=CIFAR_INFO["filename"],
            md5=CIFAR_INFO["tgz_md5"],
            remove_finished=True
        )

    assert all(
        check_integrity(CIFAR_PATH / path, md5) for path, md5 in CIFAR_INFO['file_list']
    ), "Dataset file not found or corrupted. Download it again or remove it first"

    return True


@lru_cache(maxsize=1)
def get_cifar100_cls_names() -> list[str]:

    check_and_download_cifar100()

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

    @staticmethod
    def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408),
                std=(0.2675, 0.2565, 0.2761)
            )
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408),
                std=(0.2675, 0.2565, 0.2761)
            )
        ])
        return train_transform, test_transform

    def __init__(self, images: Images, labels: Labels, transforms: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        # get image
        image = self.images[index]

        # data augmentation
        image = self.augment(image)

        return image, self.labels[index]

    def augment(self, image: torch.FloatTensor) -> torch.FloatTensor:
        return image if self.transforms is None else self.transforms(image)


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
