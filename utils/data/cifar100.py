# Standard Library
import copy
import random
import pickle
from pathlib import Path
from functools import lru_cache
from typing import Optional, Literal

# Third-Party Library
import numpy as np

# Torch Library
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

# My Library
from .denorm import DeNormalize
from ..annotation import Task, Images, Labels, ClassDataGetter


CIFAR_PATH: Path = (
    Path(__file__).resolve() / "../../../data/cifar-100-python"
).resolve()

CIFAR_INFO = {
    "url": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
    "filename": "cifar-100-python.tar.gz",
    "tgz_md5": "eb9058c3a382ffc7106e4002c42a8d85",
    "file_list": [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
        ["meta", "7973b15100ade9c7d40fb424638fde48"],
    ],
}

Split = Literal["train", "val", "test"]


@lru_cache(maxsize=1)
def check_and_download_cifar100() -> bool:
    if not CIFAR_PATH.exists():
        download_and_extract_archive(
            url=CIFAR_INFO["url"],
            extract_root=CIFAR_PATH.parent,
            download_root=CIFAR_PATH.parent,
            filename=CIFAR_INFO["filename"],
            md5=CIFAR_INFO["tgz_md5"],
            remove_finished=True,
        )

    assert all(
        check_integrity(CIFAR_PATH / path, md5) for path, md5 in CIFAR_INFO["file_list"]
    ), "Dataset file not found or corrupted. Download it again or remove it first"

    return True


# by default, whenever the cifar100.py is imported, check and download the cifar100 if it does not exists
check_and_download_cifar100()


@lru_cache(maxsize=1)
def get_cls_names() -> list[str]:

    meta = CIFAR_PATH / "meta"
    with meta.open(mode="rb") as f:
        data = pickle.load(f)

    return data["fine_label_names"]


@lru_cache(maxsize=3)
def get_data(split: Split) -> tuple[Images, Labels]:
    # sourcery skip: remove-redundant-if
    file = CIFAR_PATH / ("test" if split == "test" else "train")

    with file.open(mode="rb") as f:
        data = pickle.load(f, encoding="bytes")

    images = data[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.array(data[b"fine_labels"], dtype=np.int64)

    # these codes are not used
    if split != "test" and False:
        cls_id_mapper = {
            cls_name: cls_id for cls_id, cls_name in enumerate(get_cls_names())
        }

        cls_masks = [labels == cls_id for cls_id in cls_id_mapper.values()]

        # use first 450 as train, last 50 as validation
        start, end = (0, 450) if split == "val" else (449, -1)
        images_list, labels_list = [], []
        for mask in cls_masks:
            true_indices = np.where(mask)[0]
            mask[true_indices[start:end]] = False
            images_list.append(images[mask])
            labels_list.append(labels[mask])
        images, labels = np.concatenate(images_list), np.concatenate(labels_list)

    return images, labels


@lru_cache(maxsize=3)
def get_cls_data_getter(split: Split) -> ClassDataGetter:

    # get the cls_id
    cls_names = get_cls_names()
    cls_id_mapper = {name: id for id, name in enumerate(cls_names)}

    # read data
    images: Images
    labels: Labels
    images, labels = get_data(split)

    def cls_data_getter(cls_name: str, new_cls_id: int) -> tuple[Images, Labels]:

        # get class data
        cls_mask = labels == cls_id_mapper[cls_name]
        cls_images = images[cls_mask]
        cls_labels = np.full((cls_images.shape[0],), fill_value=new_cls_id)

        return cls_images, cls_labels

    return cls_data_getter


def get_task_data(
    split: Split, task: Task, cls_ids: list[int]
) -> tuple[Images, Labels]:

    images: list[Images]
    labels: list[Labels]
    images, labels = [], []

    cls_data_getter = get_cls_data_getter(split)

    for cls_id, cls_name in zip(cls_ids, task):
        cls_image, cls_label = cls_data_getter(cls_name, cls_id)
        images.append(cls_image)
        labels.append(cls_label)

    return np.concatenate(images), np.concatenate(labels)


def get_transforms() -> tuple[transforms.Compose, transforms.Compose, DeNormalize]:
    std = (0.2675, 0.2565, 0.2761)
    mean = (0.5071, 0.4867, 0.4408)
    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    denorm_transform = DeNormalize(mean=mean, std=std)
    return train_transform, test_transform, denorm_transform


class Cifar100Dataset(Dataset):

    def __init__(
        self,
        images: Images,
        labels: Labels,
        transforms: transforms.Compose,
        denorm_transforms: transforms.Compose,
    ) -> None:
        super().__init__()
        self.images = images
        self.labels = labels
        self.transforms = transforms
        self.denorm_transforms = denorm_transforms

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.FloatTensor, np.ndarray, torch.LongTensor]:
        # get image
        original_image: np.ndarray = self.images[index]

        # data augmentation
        augmented_image: np.ndarray = self.augment(original_image.copy())

        return augmented_image, original_image, self.labels[index]

    def augment(self, original_image: torch.FloatTensor) -> torch.FloatTensor:
        return (
            original_image
            if self.transforms is None
            else self.transforms(original_image)
        )


if __name__ == "__main__":
    import random
    from pprint import pprint
    from typing import get_args
    import matplotlib.pyplot as plt
    from ..helper import draw_image

    def test_get_cls_names():
        return get_cls_names()

    pprint(test_get_cls_names())

    def test_get_cls_data_getter(
        cls_name: str, cls_id: int, split: str, pic_num: int = 10
    ):
        assert split in get_args(Split), f"invalid {split=}"
        # get the data of a class
        cls_data_getter = get_cls_data_getter(split)
        cls_image, cls_label = cls_data_getter(cls_name, cls_id)

        print(f"{cls_image.shape=}")
        print(f"{cls_name=}, {cls_id=}, {cls_label[0]} {cls_label.shape=}")
        camel_image_tensor = torch.from_numpy(cls_image[:pic_num]).permute(0, 3, 1, 2)
        draw_image(
            camel_image_tensor,
            f"./test/get-class-data-example-{cls_name=}-{cls_id=}-{split=}-{pic_num=}.png",
        )

    test_get_cls_data_getter("apple", 0, "train", 16)
    test_get_cls_data_getter("bottle", 1, "test", 16)

    def test_get_task_data(task_len: int, split: Split):
        assert split in get_args(Split), f"invalid {split=}"
        # get the data of a task
        # fmt: off
        cls_names = get_cls_names()
        random.shuffle(cls_names)
        task = cls_names[:task_len]
        # fmt: on
        task_image, task_label = get_task_data(split, task, range(len(task)))

        num = 100 if split == "test" else 500
        print(f"{task=}")
        print(f"{task_label=}")
        print(f"{task_image.shape=}")
        task_image_tensor = torch.from_numpy(
            task_image[: num * task_len : num]
        ).permute(0, 3, 1, 2)
        draw_image(
            task_image_tensor,
            f"./test/get-task-data-test-image-{split=}-{task_len=}.png",
        )

    test_get_task_data(task_len=10, split="train")
    test_get_task_data(task_len=16, split="test")
