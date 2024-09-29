# Standard Library
import copy
from typing import Optional
from contextlib import contextmanager

# Third-Party Library
import clip

# Torch Library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.resnet import ResNet

# My Library
from utils.annotation import Task
from .base import ContinualLearningModel


class LingoCosineClassifier(nn.Module):

    def __init__(
        self, in_features: int, out_features: int, cls_embeddings: torch.FloatTensor
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.eta = nn.Parameter(torch.tensor([1.0]))
        self.class_embeddings: nn.ParameterList = nn.ParameterList([cls_embeddings])
        for param in self.class_embeddings.parameters():
            param.requires_grad = False

    def forward(self, normalized_features: torch.FloatTensor) -> torch.FloatTensor:

        class_embedding = self.get_class_embedding()
        normalized_class_embeddings = F.normalize(class_embedding, p=2, dim=-1)

        logits = F.linear(normalized_features, normalized_class_embeddings)
        scaled_output = self.eta * logits

        return logits, scaled_output

    def get_class_embedding(self) -> nn.Parameter:
        # sourcery skip: identity-comprehension
        return torch.cat([i for i in self.class_embeddings], dim=0)

    @staticmethod
    def init_from_existing_classifier(
        other: "LingoCosineClassifier",
        in_features: int,
        out_features: int,
        cls_embeddings: torch.FloatTensor,
    ) -> "LingoCosineClassifier":

        this = LingoCosineClassifier(in_features, out_features, cls_embeddings)

        this.eta = copy.deepcopy(other.eta)
        this.class_embeddings = copy.deepcopy(other.class_embeddings)
        this.class_embeddings.append(nn.Parameter(cls_embeddings))

        this.train()
        for param in this.class_embeddings.parameters():
            param.requires_grad = False

        return this


class LingoCL(nn.Module, ContinualLearningModel):

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()

        self.feature_extractor = backbone

        # set in __init__.py:get_backbone()
        self.feature_dim = self.feature_extractor.feature_dim

        # after learning the current task, the mean of exemplar sets need to be recalculated, so we need to save the current feature extractor
        self.previous_feature_extractors: list[nn.Module] = []

        self.current_classifier: LingoCosineClassifier = None
        self.previous_classifiers: list[LingoCosineClassifier] = []

        self.previous_nets: list[nn.Sequential] = []

        self.current_task: Task = None
        self.learned_classes: list[str] = []
        self.learned_tasks: list[Task] = []

        self.clip, _ = clip.load("ViT-B/32", device="cpu")

    @contextmanager
    def set_new_task(self, task: Task):
        num_cls = len(task)
        self.current_task = task

        # use clip to generate cosine similarity classifier weights
        text = clip.tokenize([f"{i}" for i in task]).to(
            next(self.feature_extractor.parameters()).device
        )

        text_features: torch.FloatTensor
        text_features = self.clip.encode_text(text)
        text_features = F.normalize(text_features, p=2, dim=1)

        # create new classifier
        self.current_classifier = LingoCosineClassifier(
            self.feature_dim, num_cls, text_features
        )

        if len(self.previous_classifiers) > 0:

            # copy the weight of last classifier
            previous_classifier = self.previous_classifiers[-1]

            self.current_classifier = (
                LingoCosineClassifier.init_from_existing_classifier(
                    previous_classifier, self.feature_dim, num_cls, text_features
                )
            )

        self.current_classifier = self.current_classifier.to(
            device=next(self.feature_extractor.parameters()).device
        )

        # return to task learning
        yield self

        # after the task, save the feature extractor and classifier
        self.previous_classifiers.append(copy.deepcopy(self.current_classifier))
        self.previous_feature_extractors.append(copy.deepcopy(self.feature_extractor))

        # freeze the current classifier and switch to evaluation mode
        self.previous_classifiers[-1].eval()
        for param in self.previous_feature_extractors[-1].parameters():
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

    def forward(
        self, image: torch.FloatTensor
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Args:
            image (torch.FloatTensor): input image, shape [B, C, H, W]
        Returns:
            torch.FloatTensor: output logits, shape [B, N] (N for number of classes learned so far)
            torch.FloatTensor: output logits multiplied by eta, i.e., scaled, shape [B, N] (N for number of classes learned so far)
        """

        # sourcery skip: inline-immediately-returned-variable
        features: torch.FloatTensor
        logits: torch.FloatTensor

        features = self.feature_extractor(image)

        normalized_features = F.normalize(
            features,
            p=2,
            dim=-1,
        )

        logits, scaled_logits = self.current_classifier(features)

        return normalized_features, logits, scaled_logits

    def get_class_embedding(self) -> nn.Parameter:
        return self.current_classifier.get_class_embedding()


if __name__ == "__main__":

    from torchvision.models import resnet34

    a = resnet34()
    a.fc = nn.Identity()
    a.feature_dim = 512
    model = LingoCL(a)
    model = model.to("cuda:0")

    train_image = torch.randn(64, 3, 32, 32).to("cuda:0")
    test_image = torch.randn(16, 3, 32, 32).to("cuda:0")

    with model.set_new_task(t := ["apple", "aquarium_fish"]):
        logits = model(train_image)

    with model.set_new_task(t := ["boy", "bottle"]):
        logits = model(train_image)
