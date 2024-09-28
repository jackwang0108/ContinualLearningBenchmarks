# Standard Library
import copy
from typing import Optional
from contextlib import contextmanager

# Third-Party Library

# Torch Library
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet

# My Library
from utils.annotation import Task
from .base import ContinualLearningModel


class iCaRL(nn.Module, ContinualLearningModel):

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()

        self.feature_extractor = backbone

        # set in __init__.py:get_backbone()
        self.feature_dim = self.feature_extractor.feature_dim

        # after learning the current task, the mean of exemplar sets need to be recalculated, so we need to save the current feature extractor
        self.previous_feature_extractors: list[nn.Module] = []

        # Note: iCaRL uses weight vectors for representation learning, not classification
        # Ref: Page 3, Architecture, Paragraph 2, Line 11: "Note that even though one can interpret these outputs as probabilities, iCaRL uses the network only for representation learning, not for the actual classification step."
        self.previous_weight_vectors: list[nn.Linear] = []
        self.current_weight_vectors: nn.Linear = None

        self.previous_nets: list[nn.Sequential] = []

        # Note: iCaRL uses Nearest-Mean-of-Exemplars to classify a given example
        self.exemplar_means: dict[str, torch.FloatTensor] = {}

        self.current_task: Task = None
        self.learned_classes: list[str] = []
        self.learned_tasks: list[Task] = []

    @contextmanager
    def set_new_task(self, task: Task):
        num_cls = len(task)
        self.current_task = task

        # before the start learning the task, expand the weight vectors

        # create new weight vectors
        self.current_weight_vectors = nn.Linear(
            in_features=self.feature_dim,
            out_features=num_cls + len(self.learned_classes),
            bias=False,
        ).to(device=next(self.feature_extractor.parameters()).device)

        # for the second task, copy the weights of last weight vectors
        if len(self.previous_weight_vectors) > 0:

            # copy the weight of previous weight vector
            previous_weight_vector = self.previous_weight_vectors[-1]

            self.current_weight_vectors.weight.data[: len(self.learned_classes), :] = (
                previous_weight_vector.weight.data[:, :]
            )

        # return to task learning
        yield self

        # after the task, save the feature extractor and weight vectors
        self.previous_weight_vectors.append(copy.deepcopy(self.current_weight_vectors))
        self.previous_feature_extractors.append(copy.deepcopy(self.feature_extractor))

        # freeze the current weight vector and switch to evaluation mode
        self.previous_weight_vectors[-1].eval()
        for param in self.previous_feature_extractors[-1].parameters():
            param.requires_grad = False

        # freeze previous feature extractor and switch to evaluation mode
        self.previous_feature_extractors[-1].eval()
        for param in self.previous_feature_extractors[-1].parameters():
            param.requires_grad = False

        self.previous_nets.append(
            nn.Sequential(
                self.previous_feature_extractors[-1],
                self.previous_weight_vectors[-1],
            )
        )

        # expand the learned classes
        self.learned_classes.extend(task)
        self.learned_tasks.append(task)

    def forward(self, image: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            image (torch.FloatTensor): input image, shape [B, C, H, W]
        Returns:
            torch.FloatTensor: output logits, shape [B, N] (N for number of classes learned so far)
        """

        # sourcery skip: inline-immediately-returned-variable
        features: torch.FloatTensor
        logits: torch.FloatTensor

        features = self.feature_extractor(image)

        logits = self.current_weight_vectors(features)

        return logits


if __name__ == "__main__":
    from utils.datasets import CLDatasetGetter
    from torch.utils.data import DataLoader

    model = iCaRL()
    model = model.to("cuda:0")

    train_image = torch.randn(64, 3, 32, 32).to("cuda:0")
    test_image = torch.randn(16, 3, 32, 32).to("cuda:0")

    with model.set_new_task(t := ["1", "2", "3"]):
        logits = model(train_image)

        # calculate mean of class
        for i, t_name in enumerate(t):
            model.exemplar_means[t_name] = (torch.ones(1, 512) * i).to("cuda:0")

        pred = model.get_preds(test_image)
