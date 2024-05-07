# Standard Library
import copy
from contextlib import contextmanager

# Third-Party Library
import clip

# Torch Library
import torch
import torch.nn as nn
import torchvision.models as models

# My Library
from .iCaRL import iCaRL
from utils.annotation import Task


class iCaRL_LingoCL(iCaRL):

    def __init__(self, feature_dim: int = 512) -> None:
        super().__init__(feature_dim)

        self.clip, _ = clip.load("ViT-B/32", device="cpu")

    @contextmanager
    def set_new_task(self, task: Task):
        num_cls = len(task)
        self.current_task = task

        # before the start learning the task, expand the weight vectors

        # create new weight vectors
        self.current_weight_vectors = nn.Linear(
            in_features=self.feature_dim, out_features=num_cls + len(self.learned_classes), bias=False).to(device=self.feature_extractor.conv1.weight.device)

        # for the second task, copy the weights of last weight vectors
        if len(self.weight_vectors) > 0:

            # copy the weight of previous weight vector
            previous_weight_vector = self.weight_vectors[-1]

            self.current_weight_vectors.weight.data[:len(
                self.learned_classes), :] = previous_weight_vector.weight.data[:, :]

        # use clip to generate weights for new classes
        text = clip.tokenize([f"a {i}" for i in task]).to(
            self.feature_extractor.conv1.weight.device)

        text_features: torch.FloatTensor
        text_features = self.clip.encode_text(text)
        text_features = text_features / \
            text_features.norm(dim=1, keepdim=True)

        # get weights for new classes
        # Ref: Page 3, Section 3.2 Our Proposed Language-Guided Supervision, (ii)
        weights = text_features

        self.current_weight_vectors.weight.data[len(
            self.learned_classes):] = weights

        # freeze the weight vector
        for param in self.current_weight_vectors.parameters():
            param.requires_grad = False

        # return to task learning
        yield self

        # after the task, save the feature extractor and weight vectors

        # switch the current weight vectors to evaluation mode
        self.current_weight_vectors.eval()

        # save the current weight vectors
        self.weight_vectors.append(self.current_weight_vectors)

        # copy the last feature extractor
        self.previous_feature_extractor = copy.deepcopy(
            self.feature_extractor)

        # freeze previous feature extractor and switch to evaluation mode
        for param in self.previous_feature_extractor.parameters():
            param.requires_grad = False

        self.previous_feature_extractor.eval()

        # expand the learned classes
        self.learned_classes.extend(task)


class CosineClassifier(nn.Module):
    def __init__(self, in_features: int, out_features: int, weights: torch.FloatTensor) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        assert out_features == weights.size(0), f"mismatched weights {out_features=} but {
            weights.size(0)=}"

        self.linear = nn.Linear(in_features=in_features,
                                out_features=out_features, bias=False)

        self.linear.weight = nn.Parameter(weights)

        # frozen parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, features: torch.FloatTensor) -> torch.FloatTensor:
        # normalized the image features for cosine similarity
        features = features / features.norm(dim=1, keepdim=True)
        return self.linear(features)


class LingoCL(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT)

        self.feature_dim = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Identity()

        self.classifiers: list[nn.Module] = []
        self.current_classifier: nn.Module = None

        self.learned_classes = []

        # Warning: clip transform is different from ours
        # load to cpu first
        self.clip, _ = clip.load("ViT-B/32", device="cpu")
        self.classifier_weights = []

    @contextmanager
    def set_new_task(self, task: list[str]):
        num_cls = len(task)

        # use clip to generate cosine similarity classifier weights

        # generate weights for current task
        text = clip.tokenize([f"a {i}" for i in task]).to(
            self.feature_extractor.conv1.weight.device)

        text_features: torch.FloatTensor
        text_features = self.clip.encode_text(text)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # get weights for all learned classes
        self.classifier_weights.append(text_features)
        weights = torch.cat(self.classifier_weights)

        self.current_classifier = CosineClassifier(
            in_features=self.feature_dim, out_features=num_cls + len(self.learned_classes), weights=weights).to(device=self.feature_extractor.conv1.weight.device)

        self.classifiers.append(self.current_classifier)

        try:
            yield self
        finally:
            self.learned_classes.extend(task)

    def forward(self, image: torch.FloatTensor) -> torch.FloatTensor:
        # sourcery skip: inline-immediately-returned-variable
        features: torch.FloatTensor
        features = self.feature_extractor(image)
        logits = self.current_classifier(features)
        return logits


if __name__ == "__main__":
    lingo = iCaRL_LingoCL()

    tasks = [["1", "2"], ["3", "4"]]

    image = torch.randn(32, 3, 32, 32)

    for task in tasks:
        with lingo.set_new_task(task):
            logits = lingo(image)

            print(logits.shape)

        for cls_name in task:
            lingo.exemplar_means[cls_name] = torch.randn(1, 512)

        preds = lingo.get_preds(image)
        print(preds)
