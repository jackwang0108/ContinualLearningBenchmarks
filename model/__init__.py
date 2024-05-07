# Standard Library

# My Library
from .iCaRL import iCaRL
from .LingoCL import *
from .finetune import Finetune

Model = Finetune | iCaRL | iCaRL_LingoCL
