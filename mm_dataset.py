
import os
import scipy.io
from typing import Optional, Any
import numpy as np
from torch_geometric.data import Data,Dataset
import torch_geometric.transforms
import torch
from torchvision.datasets.folder import default_loader
from torchvision import transforms

class MMGraphDataset(Dataset):
    """ Persistent Image dataset"""
    
    def __init__(
        self, 
        graph_path: str,
        img_path: str,
        verbosity: Optional[bool] = False,
        gnn_transform: Any = None,
        img_transform: Any = None,
        train_mode = True,
    ):
        super().__init__()
        
        self.graph_path = graph_path
        self.img_path = img_path
        
        # self.gnn_transform = gnn_transform