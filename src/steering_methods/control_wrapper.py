import torch 
import torch.nn as nn
import numpy as np
import time
import pdb
from typing import Optional, List, Dict, Any, Tuple, Union
import math
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding, apply_rotary_pos_emb
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
#from transformers.models.olmo.modeling_olmo import OLMoDecoderLayer
from transformers.models.phi.modeling_phi import PhiDecoderLayer
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from src.utils.data_utils import encode_data  # Updated import path to use data_utils from utils
import json
import pandas as pd
from tqdm import tqdm
import random
from sklearn.metrics import f1_score 
from scipy import optimize
from src.utils.config_utils import load_config
from src.utils.probe_utils_regression import LinearProbe
from transformers import AutoModelForCausalLM, AutoTokenizer


# Different strictly monotone functions 

MAP = {
    'identity': lambda x: x,
    'sigmoid': torch.nn.functional.sigmoid,
    'tanh': torch.nn.functional.tanh
}

INVERSE_MAP = {
    'identity': lambda x: x, 
    'sigmoid': lambda x: np.log(x) - np.log(1-x) if 0 < x < 1 else (- np.infty if x == 0 else np.infty),
    'tanh': np.arctanh
}




class LiSeCoBaseWrapper(torch.nn.Module):
    def __init__(self, base_layer: nn.Module, linear_probe: nn.Module, name="", device='cpu'):
        """
        W shape: d x 1
        """
        super(LiSeCoBaseWrapper, self).__init__()
        self.base_layer = base_layer
        self.device = device
        
        # Probe-related parameters
        self.probe = linear_probe.eval().half().to(device)
        self.w = linear_probe.linear.weight.detach().cpu().numpy().squeeze() # linear probe
        self.B = linear_probe.linear.bias.detach().cpu().numpy().squeeze()
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
                 linear_probe: LinearProbe, 
                 lower: float, 
                 upper: float, 
                 map_to_target_space: str, 
                 name: Optional[str] = "LiSeCo",
                 device: Optional[str] = 'cpu'
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
        super(LiSeCoWrapper, self).__init__(base_layer, linear_probe, name=name, device=device)

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
        self.forward_map = MAP[map_to_target_space]
        self.inverse_map = INVERSE_MAP[map_to_target_space]

        # Set the bounds in the "inverse space"
        self.lower = lower 
        self.upper = upper
        self.a = self.inverse_map(lower) # the thresholds we're comfortable with in target space
        self.b = self.inverse_map(upper) 
        
        assert self.a < self.b # should work because the inverse is monotone


    def eval_probe(self, x_seq, last_token_idx):
        # Get the last token representation
        last_token_rep = x_seq[torch.arange(x_seq.size(0)), last_token_idx]
        
        # Ensure probe is on the same device as input
        if self.probe.linear.weight.device != last_token_rep.device:
            self.probe = self.probe.to(last_token_rep.device)
            
        eval_result = self.probe(last_token_rep)

        # Keep as tensor until needed
        return self.forward_map(eval_result.squeeze()).detach()  # Return tensor instead of converting to numpy

    def forward(self, x, *args, **kwargs):
        t = time.time()
        x_seq, x_metadata = self.base_layer(x, *args, **kwargs)

        # Add the toxicity score to the log
        if 'position_ids' in kwargs:
            last_token_idx = kwargs['position_ids'].cpu().size(1) - 1
        else:
            last_token_idx = x_seq.size(1) - 1
            
        # Store the tensor result
        p_before = self.eval_probe(x_seq, last_token_idx)
        self.pre_adjust_toxicity_log.append(p_before)
        
        if self.control: # Make the adjustment
            # Get last token representation
            last_token_rep = x_seq[torch.arange(x_seq.size(0)), last_token_idx]
            # Get optimal theta and ensure it's on the same device
            theta = self.optimal_theta(last_token_rep)
            if theta.device != x_seq.device:
                theta = theta.to(x_seq.device)
            # Apply the adjustment
            x_seq[torch.arange(x_seq.size(0)), last_token_idx] += theta

        # Add to toxicity log
        post_adjust_p = self.eval_probe(x_seq, last_token_idx)
        if self.control: 
            assert self.lower - 0.01 <= post_adjust_p <= self.upper

        self.post_adjust_toxicity_log.append(post_adjust_p)
        self.latency.append(time.time() - t)
        return x_seq, x_metadata

    def optimal_theta(self, x):
        """Finds the optimal steering vector. Warning: ONLY WORKS FOR BATCH SIZE=1

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        theta = torch.zeros(x.shape, device=self.device, dtype=torch.float16)  # Create with half precision
        x = x.detach().cpu().numpy()

        # Classified as toxic when w.T x < a or w.T x > b.
        toxic_sequences_idx_upper = np.where(self.w @ x.T + self.B > self.b)
        toxic_sequences_idx_lower = np.where(self.w @ x.T  + self.B < self.a)
        self.toxic_sequences.append(toxic_sequences_idx_upper)
        self.toxic_sequences.append(toxic_sequences_idx_lower)

        # No intervention needed case
        if not len(toxic_sequences_idx_upper[0]) and not len(toxic_sequences_idx_lower[0]): 
            return theta

        # intervention needed
        x_toxic_upper = x[toxic_sequences_idx_upper] # index into the toxic ones only
        x_toxic_lower = x[toxic_sequences_idx_lower]

        # Upper intervention needed
        if len(x_toxic_upper):
            theta[toxic_sequences_idx_upper] = torch.tensor(
                self.w * (self.b - (self.w @ x_toxic_upper.T + self.B)) / self.w_norm**2,
                device=self.device,
                dtype=torch.float16
            )
        
        # Lower intervention needed
        if len(x_toxic_lower):
            theta[toxic_sequences_idx_lower] = torch.tensor(
                self.w * (self.a - (self.w @ x_toxic_lower.T + self.B)) / self.w_norm**2,
                device=self.device,
                dtype=torch.float16
            )

        return theta

    def reset_logs(self):
        self.toxic_sequences = []
        self.pre_adjust_toxicity_log = []
        self.post_adjust_toxicity_log = []
        self.latency = []


if __name__ == "__main__":
    # Example code for running
    # Load configuration
    config = load_config('src/config.json')
    ACCESS_TOKEN = config['access_token']
    BASE_PATH = config['base_path']

    MODEL = 'meta-llama/Meta-Llama-3-8B'
    DEVICE = 'cuda'
    EXP = 'toxicity'
    LAYERS = list(range(12, 33))
    LOWER, UPPER = 0, 0.3 # bounds on what we want p(toxic) to be
    map = 'sigmoid'

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, token=ACCESS_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(MODEL,
                                                token=ACCESS_TOKEN,
                                                load_in_8bit=True
                                                )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model.eval()

    # Load in the probe 
    Ws = [torch.load(
            f'{BASE_PATH}/experiments/{EXP}/saved_probes/{MODEL.split("/")[-1]}_linear_probe_layer_{LAYER}_rs0.pt'
            ).to(DEVICE) for LAYER in LAYERS
    ]
    [W.eval() for W in Ws]

    # EVALUATE

    # Example prompt
    prompt = "Monica Lewinsky is such a"

    # Put the probe on the layer
    layerlist = model.model.layers

    for LAYER in LAYERS:
        layerlist[LAYER-1] = LiSeCoWrapper(
                    layerlist[LAYER-1],
                    linear_probe=Ws[LAYER - 12],
                    lower=LOWER,
                    upper=UPPER,
                    map_to_target_space=map,
                    device=DEVICE
                )
    
    # Evaluate
    encoding = encode_data(tokenizer, 1, [prompt], 1, model.config.max_position_embeddings, DEVICE)[0]
    [layerlist[LAYER-1].control_on() for LAYER in LAYERS]

    outputs = model.generate(
            inputs=encoding['input_ids'], # batch size x seq len
            min_new_tokens=1,
            max_new_tokens=50,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
        )
    print('With LiSeCo:')
    generated_tokens_text = tokenizer.decode(*outputs[0], skip_special_tokens=True)
    print(generated_tokens_text)
    print('\n\n')

    # UNWRAP and evaluate baseline
    for LAYER in LAYERS:
        layerlist[LAYER-1] = layerlist[LAYER-1].base_layer

    outputs = model.generate(
            inputs=encoding['input_ids'], # batch size x seq len
            min_new_tokens=1,
            max_new_tokens=50,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
        )
    print('Original model:')
    generated_tokens_text = tokenizer.decode(*outputs[0], skip_special_tokens=True)
    print(generated_tokens_text)
