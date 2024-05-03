# Standard Library

# Third-Party Library

# Torch Library
import torch
import torch.nn.functional as F


def to_onehot(input: torch.FloatTensor, num_classes: int) -> torch.FloatTensor:
    ori_size = input.size()
    return F.one_hot(input.flatten(), num_classes=num_classes).reshape(*ori_size, -1)


def get_probas(logits: torch.FloatTensor) -> torch.FloatTensor:
    return F.softmax(logits, dim=-1)


def get_pred(probas: torch.FloatTensor) -> torch.FloatTensor:
    return probas.argmax(dim=-1)


@torch.no_grad()
def get_acc(pred: torch.FloatTensor, gt: torch.FloatTensor, num_cls: int) -> torch.FloatTensor:
    gt = to_onehot(gt, num_cls)
    pred = to_onehot(pred, num_cls)
    correct = pred * gt
    return ((correct) != 0).sum() / pred.size(0)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from .loader import CLDatasetGetter

    dataset_getter = CLDatasetGetter(
        dataset="cifar100", task_num=10, fixed_task=False)

    learned_classes = 0
    for task_id, cls_names, train_dataset, _, _ in dataset_getter:
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        image: torch.FloatTensor
        label: torch.FloatTensor

        # learn the task
        print(f"{task_id=}, {cls_names=}")
        for image, label in train_loader:
            print(image.shape)
            print(label.shape)

            onehot_label = to_onehot(
                label, dataset_getter.num_cls_per_task + learned_classes)
            print(onehot_label.shape)
            break
        learned_classes += len(cls_names)
