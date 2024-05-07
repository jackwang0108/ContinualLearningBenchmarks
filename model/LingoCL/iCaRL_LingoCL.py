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
from ..iCaRL import iCaRL
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

        features = features / features.norm(dim=1, keepdim=True)
        logits = self.current_weight_vectors(features)

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
