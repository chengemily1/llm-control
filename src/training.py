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


parser = argparse.ArgumentParser(description='training proof-of-concept')

# Data selection
parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument('--dataset_name', type=str, default='/home/echeng/llm-control/jigsaw-toxic-comment-classification-challenge')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args([])

ACCESS_TOKEN='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'


# Linear control wrapper class
class LinearControlWrapper(torch.nn.Module):
    def __init__(self, base_layer: nn.Module, linear_probe: nn.Module, name="", gamma=0.1, thres=0.05):
        """
        W shape: d x 2
        """
        super(LinearControlWrapper, self).__init__()
        self.base_layer = base_layer
        self.thres = - thres # the SVM threshold we're comfortable with

        # Probe-related parameters
        self.gamma = gamma
        self.W = linear_probe.weight # linear probe
        self.b = linear_probe.bias
        self.w = (self.W[1,:] - self.W[0,:]).detach().cpu().numpy() # as defined in algo w_2 - w_1
        self.w_norm = np.linalg.norm(self.w) # python float

    def forward(self, x, *args, **kwargs):
        print('attention_mask' in kwargs)
        print(x)
        x_seq, x_metadata = self.base_layer(x, *args, **kwargs)
        print(x_seq)
        print(kwargs)
        # TODO: why is the next iteration of generation taking a shape of 2 x 1 x d??
        # Now update the last token representation
        sequence_len, _ = torch.max(kwargs['position_ids'].cpu(), dim=1)
        pdb.set_trace()
        x_seq[torch.arange(x_seq.size(0)),sequence_len] += self.optimal_theta(
            x_seq[torch.arange(x_seq.size(0)),sequence_len] # get last token rep
        )
        print('Adjustment done')
        print(x_seq.shape)
        pdb.set_trace()
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
        pdb.set_trace()

        # Classified as toxic when (w_2 - w_1).T x < 0
        toxic_sequences_idx = np.where(x @ self.w < self.thres)
        if not len(toxic_sequences_idx[0]): 
            print('No toxic sequences')
            return theta.to(self.W.device)

        x_toxic = x[toxic_sequences_idx] # index into the toxic ones only
        
        function = lambda l: l * np.exp(l * self.w_norm**2 + x_toxic @ self.w) + l - 1/self.gamma

        # Binary search for initial condition
        x0 = np.ones(len(toxic_sequences_idx[0],)) * 0.25 / self.gamma 
        while np.all(function(x0) > 0):
            x0 = x0 * 0.5 # we know the positive root is between 0 and 1/gamma
        x0 = 2 * x0

        # print('exponent: ', exponent(x0))
        lmbda = optimize.root(function, x0, tol=1e-6) # parameter to optimize
        print(lmbda)

        assert lmbda.success == True 
        theta[toxic_sequences_idx] = torch.Tensor(
            np.expand_dims(lmbda.x, axis=1) @ np.expand_dims(self.w, axis=0)
        )
        return theta.to(self.W.device)




# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=ACCESS_TOKEN)
model = AutoModelForCausalLM.from_pretrained(args.model_name, 
                                             token=ACCESS_TOKEN,
                                             load_in_8bit=True
                                            )

if 'Llama-2' in args.model_name:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

model.eval()

# Load linear probe
W = torch.load('/home/echeng/llm-control/experiments/toxicity/linear_probe_tiny.pt').to(args.device)
W.eval()

# Replace the model layer with the control wrapper
prev_layer, model.model.layers[26] = model.model.layers[26], LinearControlWrapper(model.model.layers[26], W)
model.model.layers[26].eval()

# Load the dataset
# dataset = pd.read_csv(args.dataset_name + '/train.csv').sample(frac=0.001)
data = ['I hate that bitch.', 'I love her. She is such a '] #list(dataset['comment_text'])[:1]

def encode_data(tokenizer, N, data, batch_size, max_length, device, last_k=None):
    # last_k (int): only use the last k tokens of the input

    # If the input data is text
    if type(data[0]) == str:
        encodings = tokenizer(data, padding=True, truncation=True, max_length=max_length, return_length=True, return_tensors="pt") # output variable length encodings
        if not last_k:
            encodings = [
                {'input_ids': encodings['input_ids'][i: i + batch_size].to(device),
                'attention_mask': encodings['attention_mask'][i: i + batch_size].to(device),
                'length': encodings['length'][i: i + batch_size] }
                for i in range(0, N, batch_size)
            ]
        else:
            encodings = [
                {'input_ids': encodings['input_ids'][i: i + batch_size][-last_k:].to(device),
                'attention_mask': encodings['attention_mask'][i: i + batch_size][-last_k:].to(device) }
                for i in range(0, N, batch_size)
            ]
    else: # input data is tokens-- manually pad and batch.
        max_len = max([len(sentence) for sentence in data])
        data = [sentence for sentence in data if len(sentence) > 2]
        encodings = [tokenizer.encode(sentence[1:], padding='max_length', max_length=max_len, return_tensors="pt") \
                     for sentence in data]
        batched_encodings = [torch.stack(encodings[i: i + batch_size]).squeeze(1).to(device) for i in range(0, len(data), batch_size)]
        batched_attention_masks = [(tokens != 1).to(device).long() for tokens in batched_encodings]
        encodings = [
            {'input_ids': batched_encodings[j], 'attention_mask': batched_attention_masks[j]}
            for j in range(len(batched_encodings))
        ]

    return encodings

encodings = encode_data(tokenizer, len(data), data, args.batch_size, model.config.max_position_embeddings, args.device)

# TEST
model.eval()
outputs = model.generate(
    inputs=torch.concat([encoding['input_ids'] for encoding in encodings], axis=0), # batch size x seq len 
    # max_new_tokens=4
)
output_text = tokenizer.batch_decode(outputs)
# outputs = model.generate(
#     input_ids=encodings[0]['input_ids'], 
#     attention_mask=encodings[0]['attention_mask']
#     )
pdb.set_trace()
# 