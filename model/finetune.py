# Standard Library
import copy
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

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()

        # set in __init__.py:get_backbone()
        self.feature_dim = backbone.feature_dim

        self.feature_extractor = backbone
        self.previous_feature_extractors: list[nn.Module] = []

        self.classifier: nn.Linear = None
        self.previous_classifiers: list[nn.Linear] = []

        self.previous_nets: list[nn.Sequential] = []

        self.current_task: Task = None
        self.learned_classes: list[str] = []
        self.learned_tasks: list[Task] = []

    @contextmanager
    def set_new_task(self, task: list[str]):
        num_cls = len(task)
        self.current_task = task

        self.classifier = nn.Linear(
            in_features=self.feature_dim,
            out_features=num_cls + len(self.learned_classes),
        ).to(device=next(self.feature_extractor.parameters()).device)

        if len(self.learned_tasks) > 0:
            previous_classifier = self.previous_classifiers[-1]

            self.classifier.weight.data[: len(self.learned_classes), :] = (
                previous_classifier.weight.data[:, :]
            )

        # return to task learning
        yield self

        # after the task, save the feature extractor and classifiers
        self.previous_classifiers.append(copy.deepcopy(self.classifier))
        self.previous_feature_extractors.append(copy.deepcopy(self.feature_extractor))

        # freeze the current weight vector and switch to evaluation mode
        self.previous_classifiers[-1].eval()
        for param in self.previous_classifiers[-1].parameters():
            param.requires_grad = False

        # freeze previous feature extractor and switch to evaluation mode
        self.previous_feature_extractors[-1].eval()
        for param in self.previous_feature_extractors[-1].parameters():
            param.requires_grad = False

        self.previous_nets.append(
            nn.Sequential(
                self.previous_feature_extractors[-1],
                self.previous_classifiers[-1],
            )
        )

        # expand the learned classes
        self.learned_classes.extend(task)
        self.learned_tasks.append(task)

    def forward(self, image: torch.FloatTensor) -> torch.FloatTensor:
        # sourcery skip: inline-immediately-returned-variable
        features: torch.FloatTensor
        features = self.feature_extractor(image)
        logits = self.classifier(features)
        return logits


if __name__ == "__main__":
    from utils.datasets import CLDatasetGetter
    from torch.utils.data import DataLoader

    model = Finetune()
    model = model.to("mps")
