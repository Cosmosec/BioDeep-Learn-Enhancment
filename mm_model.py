import torch
from torch import nn
from typing import List, Optional, Tuple
from models import MLP, GCNResNet
from utils import init_weights
from const import(
    GRAPH,
    IMAGE,
    LABEL,
    LOGITS, 
    FEATURES,
    GNN_MODEL,
    TIMM_MODEL,
    FUSION_MLP,
    FUSION_TRANSFORMER,
    GNN_TRANSFORMER,
    GNN_RESNET,
)

from models import gnn_model_dict,fusion_dict
import timm 
import functools
from ft_transformer import FT_Transformer

class TIMM(nn.Module):
    def __init__(
            self,
            prefix: str,
            model_name: str,
      