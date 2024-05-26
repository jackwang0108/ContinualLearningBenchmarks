# Standard Library
from contextlib import contextmanager

# Third-Party Library

# Torch Library
import torch
import torch.nn as nn
import torchvision.models as models

# My Library
from utils.annotation import Task
from .base import ContinualLearningModel


class Finetune(nn.Module, ContinualLearningModel):

    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT)

        self.feature_dim = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Identity()

        self.classifiers: list[nn.Linear] = []
        self.current_classifier: nn.Linear = None

        self.current_task: Task = None
        self.learned_classes: list[str] = []

    @contextmanager
    def set_new_task(self, task: list[str]):
        num_cls = len(task)
        self.current_task = task

        self.current_classifier = nn.Linear(
            in_features=self.feature_dim, out_features=num_cls + len(self.learned_classes)).to(device=self.feature_extractor.conv1.weight.device)

        if len(self.classifiers) != 0:
            previous_classifier = self.classifiers[-1]

            self.current_classifier.weight.data[:len(
                self.learned_classes), :] = previous_classifier.weight.data[:, :]

        self.classifiers.append(self.current_classifier)

        # return to task learning
        yield self

        # expand the learned classes
        self.learned_classes.extend(task)

    def forward(self, image: torch.FloatTensor) -> torch.FloatTensor:
        # sourcery skip: inline-immediately-returned-variable
        features: torch.FloatTensor
        features = self.feature_extractor(image)
        logits = self.current_classifier(features)
        return logits


if __name__ == "__main__":
    from utils.datasets import CLDatasetGetter
    from torch.utils.data import DataLoader

    model = Finetune()
    model = model.to("mps")
