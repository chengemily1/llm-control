import torch 
import torch.nn as nn
import pandas as pd
import numpy as np
import pdb
import argparse

class ActAddWrapper(torch.nn.Module):

    def __init__(self, base_layer: nn.Module, 
                 layer_steer: torch.Tensor, c=3):
        """
        W shape: d x 2
        """
        super(ActAddWrapper, self).__init__()
        self.base_layer = base_layer
        self.layer_steer = layer_steer
        self.c = c

    def forward(self, x, *args, **kwargs):
        x_seq, x_metadata = self.base_layer(x, *args, **kwargs)

        # Add at the first token as in App B of the paper.        
        x_seq[torch.arange(x_seq.size(0)),0] += self.c * self.layer_steer

        return x_seq, x_metadata
