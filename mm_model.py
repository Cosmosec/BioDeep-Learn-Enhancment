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
            num_classes: Optional[int] = 0,
            pretrained: Optional[bool] = True,
    ):
        super(TIMM, self).__init__()
        self.prefix = prefix 
        
        # self.data_key = f"{prefix}_{IMAGE}"
        # self.label_key = f"{prefix}_{LABEL}"
        
        self.data_key = f"{IMAGE}"
        self.label_key = f"{LABEL}"
        
        self.num_classes = num_classes
        
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0
        )
        
        self.out_features = self.model.num_features
        
        self.head = nn.Linear(self.out_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head.apply(init_weights)
        
        
     
    def forward(
        self, 
        batch
    ):
        data = batch[self.data_key]
        
        features = self.model(data)
        logits = self.head(features)
            
        return {
            self.prefix: {
                LOGITS: logits,
                FEATURES: features,
            }
        }     
        

class GNN(nn.Module):
    def __init__(
            self,
            prefix: str,
            model_name: str,
            num_classes: int,
            in_features: int, 
            hidden_features: Optional[int] = 256, 
            out_features: Optional[int] = 256,
            pooling: Optional[float] = 0.5,
            activation: Optional[str] = "gelu",
    ):
        
        super(GNN, self).__init__()
        
        self.prefix = prefix 
        self.num_classes = num_classes
        
        # self.data_key = f"{prefix}_{GRAPH}"
        # self.label_key = f"{prefix}_{LABEL}"
        
        self.data_key = f"{GRAPH}"
        self.label_key = f"{LABEL}"
        
        self.model 