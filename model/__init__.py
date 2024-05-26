# Standard Library

# My Library
from .iCaRL import iCaRL
from .LUCIR import LUCIR
from .finetune import Finetune
from .LingoCL.LingoCL import LingoCL
from .LingoCL.iCaRL_LingoCL import iCaRL_LingoCL

Model = Finetune | iCaRL | LUCIR | LingoCL | iCaRL_LingoCL

__all__ = [
    # Typing
    "Model",
    "Finetune",
    "iCaRL",
    "LUCIR",
    "LingoCL"
]
