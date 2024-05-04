# Standard Library

# My Library
from .LingoCL import LingoCL
from .finetune import Finetune

Model = Finetune | LingoCL
