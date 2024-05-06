# Standard Library

# My Library
from .iCaRL import iCaRL
from .LingoCL import LingoCL
from .finetune import Finetune

Model = Finetune | LingoCL
