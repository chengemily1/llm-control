import argparse
import torch
from transformers import AutoModelForCausalLM, GPTNeoXTokenizerFast, AutoTokenizer
from datasets import load_dataset
import seaborn as sns
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from data_util import encode_data

parser = argparse.ArgumentParser(description='ID computation')

# Data selection
parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--experiment', default='sentiment')
args = parser.parse_args()
print(args)

ACCESS_TOKEN='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'

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name, 
                                          token=ACCESS_TOKEN,
                                          trust_remote_code=True,
                                          )
model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                             token=ACCESS_TOKEN,
                                             load_in_8bit=True,
                                             trust_remote_code=True
                                            )

if 'Llama' in args.model_name:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
elif 'pythia' in args.model_name or 'mistral' in args.model_name:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()

if args.experiment == 'sentiment':
    source, target = ('optimism', 'despair')
elif args.experiment == 'toxicity':
    source, target = ('toxicity', 'kindness')
else:
    source, target = ('formal', 'casual')

# Extract reps at every layer
def last_token_rep(x, attention_mask, padding='right'):
    seq_len = attention_mask.sum(dim=1)
    indices = (seq_len - 1)
    last_token_rep = x[torch.arange(x.size(0)), indices] if padding=='right' else x[torch.arange(x.size(0)), -1]
    return last_token_rep.cpu()


# PROCESS AND SAVE REPS
with torch.no_grad():
    representations = []
    for pt in source, target:
        encoding = encode_data(tokenizer, 1, [pt], 1, model.config.max_position_embeddings, args.device)[0]
        output = model(encoding['input_ids'], attention_mask=encoding['attention_mask'], output_hidden_states=True)['hidden_states']
        pooled_output = tuple([last_token_rep(layer, encoding['attention_mask'], padding=tokenizer.padding_side) for layer in output])
        representations.append(pooled_output)
    representations = [list(batch) for batch in zip(*representations)]
    representations = [torch.cat(batches, dim=0) for batches in representations]
    print('Layer 1 reps shape: ')
    print(representations[1].shape)

    steers = []
    for representation in representations:
        steer = representation[1,:] - representation[0,:] # target - source
        steers.append(steer)
    print(len(steers))
    torch.save(steers, f'/home/echeng/llm-control/experiments/{args.experiment}/saved_layersteers/{args.model_name.split("/")[-1]}_steers.pt')
