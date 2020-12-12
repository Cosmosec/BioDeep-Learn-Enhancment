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
        
        self.model = gnn_model_dict[model_name](
            in_features = in_features,
            hidden_features = hidden_features,
            out_features = out_features,
            pooling = pooling,
            activation = activation,
        )
        
        self.head = nn.Linear(out_features, num_classes) if num_classes > 0 else nn.Identity()
        self.out_features = out_features
        
        
    def forward(self, batch):
        data = batch[self.data_key]
        
        features = self.model(data)
        logits = self.head(features)
            
        return {
            self.prefix: {
                LOGITS: logits,
                FEATURES: features,
            }
        }



class FusionMLP(nn.Module):
    def __init__(
        self,
        prefix: str,
        models: list,
        num_classes: int,
        hidden_features: List[int],
        adapt_in_features: Optional[str] = None,
        activation: Optional[str] = "gelu",
        dropout_prob: Optional[float] = 0.5,
        normalization: Optional[str] = "layer_norm",
    ):
    
        super().__init__()
        self.prefix = prefix
        self.model = nn.ModuleList(models)   
        
        # TODO: Add out_features to each model
        raw_in_features = [per_model.out_features for per_model in models]
        
        if adapt_in_features is not None:
            if adapt_in_features == "min":
                base_in_feat = min(raw_in_features)
            elif adapt_in_features == "max":
                base_in_feat = max(raw_in_features)
            else:
                raise ValueError(f"unknown adapt_in_features: {adapt_in_features}")

            self.adapter = nn.ModuleList(
                [nn.Linear(in_feat, base_in_feat) for in_feat in raw_in_features]
            )
            
            in_features = base_in_feat * len(raw_in_features)
        else:
            self.adapter = nn.ModuleList(
                [nn.Identity() for _ in range(len(raw_in_features))]
            )
            in_features = sum(raw_in_features)

        assert len(self.adapter) == len(self.model)
        
        
        fusion_mlp = []
        for per_hidden_features in hidden_features:
            fusion_mlp.append(
                MLP(
                    in_features=in_features,
                    hidden_features=per_hidden_features,
                    out_features=per_hidden_features,
                    num_layers=1,
                    activation=activation,
                    dropout_prob=dropout_prob,
                    normalization=normalization,
                )
            )
            in_features = per_hidden_features
            
        self.fusion_mlp = nn.Sequential(*fusion_mlp)
        # in_features has become the latest hidden size
        self.head = nn.Linear(in_features, num_classes)
        # init weights
        
        self.adapter.apply(init_weights)
        self.fusion_mlp.apply(init_weights)
        self.head.apply(init_weights)
        
    def forward(
        self,
        batch: dict,
    ):
        multimodal_features = []

        for per_model, per_adapter in zip(self.model, self.adapter):
            per_output = per_model(batch)
            multimodal_features.append(per_adapter(per_output[per_model.prefix][FEATURES]))
            

        features = self.fusion_mlp(torch.cat(multimodal_features, dim=1))
        logits = self.head(features)
        # fusion_output = {
        #     self.prefix: {
        #         LOGITS: logits,
        #         FEATURES: features,
        #     }
        # }
        
        # return fusion_output
        return logits

  

class FusionTransformer(nn.Module):

    def __init__(
            self,
            prefix: str,
            models: list,
            hidden_features: int,
            num_classes: int,
            adapt_in_features: Optional[str] = None,
    ):
        super().__init__()
        
        self.model = nn.ModuleList(models)

        raw_in_features = [per_model.out_features for per_model in models]
        if adapt_in_features is not None:
            if adapt_in_features == "min":
                base_in_feat = min(raw_in_features)
            elif adapt_in_features == "max":
                base_in_feat = max(raw_in_features)
            else:
                raise ValueError(f"unknown adapt_in_features: {adapt_in_features}")

            self.adapter = nn.ModuleList(
                [nn.Linear(in_feat, base_in_feat) for in_feat in raw_in_features]
            )

            in_features = base_in_feat
        else:
            self.adapter = nn.ModuleList(
                [nn.Identity() for _ in range(len(raw_in_features))]
          