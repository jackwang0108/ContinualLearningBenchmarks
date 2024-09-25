# Standard Library
import copy
from typing import Optional
from contextlib import contextmanager

# Third-Party Library

# Torch Library
import torch
import torch.nn as nn

# My Library
from utils.annotation import Task


class ContinualLearningModel:
    """
    Abstract Base Class for all Continual Learning model.

    ContinualLearningModel defines several methods that must be overwritten by sub-class:
        - set_new_task


    Raises:
        NotImplementedError: if the required methods are not implemented by the sub-class
    """
    
    feature_dim: int
    feature_extractor: nn.Module
    

    @contextmanager
    def set_new_task(self, task: Task):
        raise NotImplementedError

    def forward(self, image: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError

    def get_preds(self, image: torch.FloatTensor, top: int = 1) -> torch.FloatTensor:
        raise NotImplementedError
