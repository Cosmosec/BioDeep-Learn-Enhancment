
'''
codes borrowed from AutoMMPredictor
'''
from torch import nn
from typing import Optional

ALL_ACT_LAYERS = {
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}


class Unit(nn.Module):
    """
    One MLP layer. 
    It orders the operations as: norm -> fc -> act_fn -> dropout
    """

    def __init__(
            self,
            normalization: str,
            in_features: int,
            out_features: int,
            activation: str,
            dropout_prob: float,
    ):
        """
        Parameters
        ----------
        normalization
            Name of activation function.
        in_features
            Dimension of input features.
        out_features
            Dimension of output features.
        activation
            Name of activation function.
        dropout_prob
            Dropout probability.
        """
        super().__init__()
        if normalization == "layer_norm":
            self.norm = nn.LayerNorm(in_features)