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


# Linear control wrapper class
class LinearControlWrapper(torch.nn.Module):
    def __init__(self, base_layer: nn.Module, 
                 linear_probe: nn.Module, name="", p=0.4):
        """
        W shape: d x 2
        """
        super(LinearControlWrapper, self).__init__()
        self.base_layer = base_layer
        assert (0 < p) and (p < 1)
        self.p = p # the SVM threshold we're comfortable with

        # Probe-related parameters
        self.probe = linear_probe.eval().half()
        self.W = linear_probe.weight # linear probe
        self.w1 = self.W[0,:].detach().cpu().numpy()
        self.w2 = self.W[1,:].detach().cpu().numpy()
        self.b = linear_probe.bias
        self.w = self.w1 - self.w2 # as defined in algo w_1 - w_2
        self.w_norm = np.linalg.norm(self.w) # python float

        # Logging
        self.toxic_sequences = []
        self.pre_adjust_toxicity_log = []
        self.post_adjust_toxicity_log = []

        # Control off or on
        self.control = False

    def control_off(self):
        self.control = False

    def control_on(self):
        self.control = True

    def eval_probe(self, x_seq, last_token_idx):
        eval_result = self.probe(x_seq[torch.arange(x_seq.size(0)),last_token_idx])
        return nn.functional.softmax(eval_result, dim=-1)[:,1].detach().cpu().numpy() # this is the probscore

    def forward(self, x, *args, **kwargs):
        x_seq, x_metadata = self.base_layer(x, *args, **kwargs)

        # Add the toxicity score to the log
        last_token_idx = kwargs['position_ids'].cpu().size(1) - 1
        self.pre_adjust_toxicity_log.append(self.eval_probe(x_seq, last_token_idx))
        
        if self.control: # Make the adjustment
            x_seq[torch.arange(x_seq.size(0)),last_token_idx] += self.optimal_theta(
                x_seq[torch.arange(x_seq.size(0)),last_token_idx] # get last token rep
            )

        # Add to toxicity log
        self.post_adjust_toxicity_log.append(self.eval_probe(x_seq, last_token_idx)) # this is the probscore

        return x_seq, x_metadata

    def optimal_theta(self, x):
        """Finds the optimal steering vector.

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        theta = torch.zeros(x.shape) # batch size x d
        x = x.detach().cpu().numpy()

        # Classified as toxic when (w_1 - w_2).T x < log (1/p - 1)
        toxic_sequences_idx = np.where(x @ self.w < np.log(1 / self.p - 1))

        self.toxic_sequences.append(toxic_sequences_idx)

        if not len(toxic_sequences_idx[0]): 
            print('no intervention needed')
            return theta.to(self.W.device)

        print('intervention needed')
        print(f"Sequence is toxic at token {len(self.pre_adjust_toxicity_log)}")

        x_toxic = x[toxic_sequences_idx] # index into the toxic ones only
        pdb.set_trace()
        theta[toxic_sequences_idx] = - self.w * (x_toxic @ self.w - np.log(1/self.p - 1)) / self.w_norm**2

        return theta.to(self.W.device)        
        # Binary search for initial condition
        x0 = np.ones(size,) * min(0.01, 0.25 * 1/self.gamma) 
        while np.all(root_function(x0) > 0):
            x0 = x0 * 0.5 # we know the positive root is between 0 and 1/gamma
        x0 = 2 * x0

        lmbda = optimize.root(root_function, x0, tol=1e-6) # parameter to optimize
        print(lmbda)

        assert lmbda.success == True 
        print('VECTOR NORM: ', np.linalg.norm(lmbda.x * self.w))
        pdb.set_trace()
        return lmbda

    def reset_logs(self):
        self.toxic_sequences = []
        self.pre_adjust_toxicity_log = []
        self.post_adjust_toxicity_log = []

