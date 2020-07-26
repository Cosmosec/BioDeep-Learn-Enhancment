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
    F