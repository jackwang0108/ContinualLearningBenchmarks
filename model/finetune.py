# Standard Library
from contextlib import contextmanager

# Third-Party Library

# Torch Library
import torch
import torch.nn as nn
import torchvision.models as models


class Finetune(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT)

        self.feature_dim = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Identity()

        self.classifiers: list[nn.Module] = []
        self.current_classifier: nn.Module = None

        self.learned_classes = []

    @contextmanager
    def set_new_task(self, task: list[str]):
        num_cls = len(task)

        self.current_classifier = nn.Linear(
            in_features=self.feature_dim, out_features=num_cls + len(self.learned_classes)).to(device=self.feature_extractor.conv1.weight.device)

        if len(self.classifiers) != 0:
            previous_classifier = self.classifiers[-1]

            self.current_classifier.weight.data[:len(
                self.learned_classes), :] = previous_classifier.weight.data[:, :]

        self.classifiers.append(self.current_classifier)

        try:
            yield self
        finally:
            self.learned_classes.extend(task)

    def forward(self, image: torch.FloatTensor) -> torch.FloatTensor:
        # sourcery skip: inline-immediately-returned-variable
        features = self.feature_extractor(image)
        logits = self.current_classifier(features)
        return logits


if __name__ == "__main__":
    from utils.loader import CLDatasetGetter
    from torch.utils.data import DataLoader

    model = Finetune()
    print(f"{model.feature_dim=}")

    dataset_getter = CLDatasetGetter(
        dataset="cifar100", task_num=10, fixed_task=False)

    for task_id, current_task, test_classes, train_dataset, val_dataset, test_dataset in dataset_getter:
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        image: torch.FloatTensor
        label: torch.FloatTensor

        model = model.set_new_task(current_task)

        # learn the task
        for image, label in train_loader:
            print(image.shape)
            print(label.shape)

            pred = model(image)

            print(f"{task_id=}, {pred.shape=}")

            break
