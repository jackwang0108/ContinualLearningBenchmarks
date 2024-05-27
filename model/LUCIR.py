import copy
from typing import Optional
from contextlib import contextmanager

# Third-Party Library

# Torch Library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.resnet import ResNet

# My Library
from utils.annotation import Task
from .base import ContinualLearningModel


class LUCIR(nn.Module, ContinualLearningModel):
    def __init__(self, feature_dim: int = 512) -> None:
        super().__init__()

        self.feature_extractor: ResNet = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT
        )

        # map the features from feature space into prototype space
        self.feature_extractor.fc = nn.Sequential(
            nn.Linear(self.feature_extractor.fc.in_features, feature_dim)
        )

        # Note: LUCIR uses cosine classifiers
        # Ref: Page 4, 3.2. Cosine Normalization
        self.eta = nn.Parameter(torch.ones([1]))
        self.weight_vectors: list[nn.Parameter] = []
        self.current_weight_vectors: nn.Parameter = None

        self.feature_dim = feature_dim

        self.current_task: Task = None
        self.learned_classes: list[str] = None

    @contextmanager
    def set_new_task(self, task: Task):
        num_cls = len(task)
        self.current_task = Task

        # before the start learning the task, expand the weight vectors
        self.current_weight_vectors = torch.zeros(
            num_cls + len(self.learned_classes), self.feature_dim)

        # for the second task, copy the weights of last weight vectors
        if len(self.learned_classes) > 0:

            # copy the weight of previous weight vector
            previous_weight_vectors = self.weight_vectors[-1]

            self.current_weight_vectors.data[:len(
                self.learned_classes)] = previous_weight_vectors.data[:, :]

        self.current_weight_vectors = nn.Parameter(self.current_weight_vectors)

        # return to task learning
        yield self

        # after the task, save the feature extractor and weight vectors

        # freeze the current weight vector
        self.current_weight_vectors.requires_grad = False

        self.weight_vectors.append(self.current_weight_vectors)

        # expand the learned classes
        self.learned_classes.extend(task)

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

        normalized_features = F.normalize(
            features, p=2, dim=1
        )

        normalized_weight_vectors = F.normalize(
            self.current_weight_vectors, p=2, dim=1
        )

        logits = F.linear(
            normalized_features, normalized_weight_vectors
        ) * self.eta

        return logits

    @torch.no_grad()
    def get_preds(self, image: torch.FloatTensor, top: int = 1) -> torch.IntTensor:
        logits = self.forward(image)
        return logits.topk(k=top, dim=1)[1]


if __name__ == "__main__":
    from utils.datasets import CLDatasetGetter
    from torch.utils.data import DataLoader

    model = LUCIR()
    model = model.to("cuda:0")

    train_image = torch.randn(64, 3, 32, 32).to("cuda:0")
    test_image = torch.randn(16, 3, 32, 32).to("cuda:0")

    with model.set_new_task(t := ["1", "2", "3"]):
        logits = model(train_image)

        # calculate mean of class
        for i, t_name in enumerate(t):
            model.exemplar_means[t_name] = (
                torch.ones(1, 512) * i).to("cuda:0")

        pred = model.get_preds(test_image)
