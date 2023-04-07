import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange

# helper functions

def exists(val):
    return val is not None

# classes