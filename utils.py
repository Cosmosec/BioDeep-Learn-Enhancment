
from torch import nn
from omegaconf import OmegaConf, DictConfig
from typing import Optional, List, Any, Dict, Tuple, Union
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt 

def init_weights(module: nn.Module):
    """
    Initialize one module. It uses xavier_norm to initialize nn.Embedding
    and xavier_uniform to initialize nn.Linear's weight.

    Parameters
    ----------
    module
        A Pytorch nn.Module.
    """
    if isinstance(module, nn.Embedding):
        nn.init.xavier_normal_(module.weight)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def ema_model_parameter_ini(model,ema_model):
    for param_main, param_ema in zip(model.parameters(), ema_model.parameters()):  
        param_ema.data.copy_(param_main.data)  
        param_ema.requires_grad = False  
    # for param_ema in ema_model.parameters():
    #     param_ema.detach_()
    # ema_model.eval()
    return ema_model

def ema_model_parameter_update(model,ema_model,theta):
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        ema_param.data.mul_(1-theta).add_(param.data, alpha = theta)
    return ema_model



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
