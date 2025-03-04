import argparse
import torch 
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import pdb
import random
from sklearn.metrics import f1_score 
from scipy import optimize  
import time
from typing import Optional

# Different strictly monotone functions 

MAP = {
    'identity': lambda x: x,
    'sigmoid': torch.nn.functional.sigmoid,
    'tanh': torch.nn.functional.tanh
}

INVERSE_MAP = {
    'identity': lambda x: x, 
    'sigmoid': lambda x: np.log(1/x - 1),
    'tanh': np.arctanh
}




class LiSeCoBaseWrapper(torch.nn.Module):
    def __init__(self, base_layer: nn.Module, linear_probe: nn.Module, name=""):
        """
        W shape: d x 1
        """
        super(LiSeCoBaseWrapper, self).__init__()
        self.base_layer = base_layer
        
        # Probe-related parameters
        self.probe = linear_probe.eval().half()
        self.w = linear_probe.weight # linear probe
        self.b = linear_probe.bias
        self.w_norm = np.linalg.norm(self.w) # python float

        # Logging
        self.toxic_sequences = []
        self.pre_adjust_toxicity_log = []
        self.post_adjust_toxicity_log = []
        self.latency = []

        # Control off or on
        self.control = False

    def control_off(self):
        self.control = False

    def control_on(self):
        self.control = True

    def reset_logs(self):
        self.toxic_sequences = []
        self.pre_adjust_toxicity_log = []
        self.post_adjust_toxicity_log = []
        self.latency = []



# Linear control wrapper class
class LiSeCoWrapper(LiSeCoBaseWrapper):
    def __init__(self, base_layer: nn.Module, 
                 linear_probe: nn.Module, 
                 lower: float, 
                 upper: float, 
                 map_to_target_space: str, 
                 name: Optional[str] = "LiSeCo"
                 ):
        """Sets up a layer wrapper to constrain the score, as determined by the linear probe, to [lower, upper] in 
        target space.

        Args:
            base_layer (nn.Module): original layer of the model.
            linear_probe (nn.Module): in R^{dx1}, pre-trained linear module.
            lower (float): lower bound in R, in the target space
            upper (float): upper bound in R, in the target space
            map_to_target_space (str): name of map to target space after the linear transformation. supported: sigmoid, tanh, identity
            name (Optional[str], optional): name the layer. Defaults to "LiSeCo".
        """
        # This handles probe-related parameters
        super(LiSeCoWrapper, self).__init__(base_layer, linear_probe, name=name)

        # Checks
        try:
            assert lower < upper 
        except:
            print('Error: the lower bound needs to be strictly less than the upper bound.')

        # sigmoid check
        if map_to_target_space == 'sigmoid':
            try:
                assert 0 <= lower <= 1
                assert 0 <= upper <= 1
            except:
                print('Error: make sure lower and upper are in probability space [0,1].')

        # Set the map
        self.forward_map = MAP(map_to_target_space)
        self.inverse_map = INVERSE_MAP(map_to_target_space)

        # Set the bounds in the "inverse space"
        self.a = self.inverse_map(lower) # the thresholds we're comfortable with in target space
        self.b = self.inverse_map(upper) 
        assert self.a < self.b # should work because the inverse is monotone


    def eval_probe(self, x_seq, last_token_idx):
        eval_result = self.probe(x_seq[torch.arange(x_seq.size(0)),last_token_idx])
        return self.forward_map(eval_result, dim=-1)[:,1].detach().cpu().numpy() # this is the score in target space

    def forward(self, x, *args, **kwargs):
        t = time.time()
        x_seq, x_metadata = self.base_layer(x, *args, **kwargs)

        # Add the toxicity score to the log
        if 'position_ids' in kwargs:
            last_token_idx = kwargs['position_ids'].cpu().size(1) - 1
        else:
            last_token_idx = x_seq.size(1) - 1
            
        self.pre_adjust_toxicity_log.append(self.eval_probe(x_seq, last_token_idx))
        
        if self.control: # Make the adjustment
            x_seq[torch.arange(x_seq.size(0)),last_token_idx] += self.optimal_theta(
                x_seq[torch.arange(x_seq.size(0)),last_token_idx] # get last token rep
            )

        # Add to toxicity log
        self.post_adjust_toxicity_log.append(self.eval_probe(x_seq, last_token_idx)) # this is the probscore
        self.latency.append(time.time() - t)
        return x_seq, x_metadata

    def optimal_theta(self, x):
        """Finds the optimal steering vector. Warning: ONLY WORKS FOR BATCH SIZE=1

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        theta = torch.zeros(x.shape) # batch size x d
        x = x.detach().cpu().numpy()

        # Classified as toxic when w.T x < a or w.T x > b.
        toxic_sequences_idx_upper = np.where(x @ self.w > self.b)
        toxic_sequences_idx_lower = np.where(x @ self.w < self.a)
        self.toxic_sequences.append(toxic_sequences_idx_upper)
        self.toxic_sequences.append(toxic_sequences_idx_lower)

        # No intervention needed case
        if not len(toxic_sequences_idx_upper[0]) and not len(toxic_sequences_idx_lower[0]): 
            return theta.to(self.W.device)

        # intervention needed
        print(f"Sequence is toxic at token {len(self.pre_adjust_toxicity_log)}")

        x_toxic_upper = x[toxic_sequences_idx_upper] # index into the toxic ones only
        x_toxic_lower = x[toxic_sequences_idx_lower]

        # Upper intervention needed
        if len(x_toxic_upper):
            theta[toxic_sequences_idx_upper] = torch.FloatTensor(self.w * (self.b - x_toxic_upper @ self.w) / self.w_norm**2)
        
        # Lower intervention needed
        if len(x_toxic_lower):
            theta[toxic_sequences_idx_lower] = torch.FloatTensor(self.w * (self.a - x_toxic_lower @ self.w) / self.w_norm**2)

        return theta.to(self.W.device)        

    def reset_logs(self):
        self.toxic_sequences = []
        self.pre_adjust_toxicity_log = []
        self.post_adjust_toxicity_log = []
        self.latency = []

