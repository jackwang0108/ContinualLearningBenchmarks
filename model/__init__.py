# Standard Library
import warnings
from typing import Literal, get_args

# Torch Library
import torch
import torch.nn as nn
import torchvision.models as models

# My Library
from .iCaRL import iCaRL
from .LUCIR import LUCIR
from .finetune import Finetune
from .LingoCL import LingoCL

Model = Finetune | iCaRL | LUCIR | LingoCL

__all__ = [
    # Typing
    "Model",
    "Finetune",
    "iCaRL",
    "LUCIR",
    "LingoCL"
]

AvailableBackbone = Literal[
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
]
available_backbone = get_args(AvailableBackbone)


def get_backbone(backbone: AvailableBackbone, pretrained: bool=False) -> nn.Module:
    assert backbone in available_backbone, f"unknown backbone: {backbone}, available backbones are {available_backbone}"
    if backbone.startswith("resnet"):
        
        weights = None
        if pretrained:
            weights = getattr(models, f"ResNet{backbone.replace('resnet', "")}_Weights").DEFAULT
            
            warnings.warn("normally, you shouldn't use a pretrained feature extractor for it contains prior knowledge acquired from other dataset like imagenet, which may lead to an unfair comparison of continual learning ability.")
            
        feature_extractor: models.ResNet = getattr(models, backbone)(weights=weights)
        
        # remove the fc layers
        feature_extractor.feature_dim = feature_extractor.fc.in_features
        feature_extractor.fc = nn.Identity()
        
        # the original first convolution kernel (7,7) in resnet is too large, replace it with a smaller one if not using pretrained arguments
        if pretrained:
            warnings.warn("the original first convolution kernel (7,7) in resnet is too large, but since you use pretrained backbone it is remained. This might cause some error for tiny images forward inference and loses accuracies")
        else:
            feature_extractor.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        
        # when input batched image is [1, C, H, W], resnet will be wrong for Resnet._forward_impl.layer4(x)
        # this is because the maxpool layer. So, remove the maxpool
        # feature_extractor.maxpool = nn.Identity()
    return feature_extractor
