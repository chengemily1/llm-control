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

from control_wrapper import LinearControlWrapper
from actadd_wrapper import ActAddWrapper
from instructions import *
from data_util import encode_data

random.seed(42)

parser = argparse.ArgumentParser(description='training proof-of-concept')

# Data selection
parser.add_argument('--experiment', type=str, default='toxicity')
parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument('--method', default='baseline', choices=['baseline', 'ours', 'actadd', 'instruct'])
parser.add_argument('--layers', metavar='N', type=int, nargs='+',
                        help='an integer or a list of integers')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--p', type=float, default=0.3)
parser.add_argument('--c', help="Actadd intervention strength", type=float, default=3)
parser.add_argument('--l', default=6, type=int)
args = parser.parse_args()

args.dataset_name = f'/home/echeng/llm-control/experiments/test_toxicity.csv' # last minute switch

ACCESS_TOKEN='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'

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=ACCESS_TOKEN)
model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                             token=ACCESS_TOKEN,
                                             load_in_8bit=True
                                            )

if 'Llama' in args.model_name:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
elif 'pythia' in args.model_name or 'mistral' in args.model_name:
    tokenizer.pad_token = tokenizer.eos_token
elif 'OLMo' in args.model_name:
    model.config.max_position_embeddings = model.config.max_sequence_length


model.eval()
# pdb.set_trace()

# Load all linear probes
num_layers = model.config.num_hidden_layers
Ws = [
    torch.load(
        f'/home/echeng/llm-control/experiments/{args.experiment}/saved_probes/{args.model_name.split("/")[-1]}_linear_probe_layer_{layer}.pt'
        ).to(args.device)
    for layer in range(1, num_layers+1)
]
[W.eval() for W in Ws]

# Load the dataset
dataset = pd.read_csv(args.dataset_name)
text_field = 'prompt'
data = list(dataset[text_field])

if args.method == 'instruct':
    data = transform_dataset(data)

# ORIGINAL BASELINE
model.eval()
print(args)
# pdb.set_trace()

if 'opt' in args.model_name:
    layerlist = model.model.decoder.layers
elif 'pythia' in args.model_name:
    layerlist = model.gpt_neox.layers
else:
    layerlist = model.model.layers

###################### ACTADD
if args.method == 'actadd':
    # load the layersteers
    steers = torch.load(
        f'/home/echeng/llm-control/experiments/{args.experiment}/saved_layersteers/{args.model_name.split("/")[-1]}_steers.pt'
        )[1:]
    # wrap the layers
    layerlist[args.l] = ActAddWrapper(layerlist[args.l], steers[args.l].to(args.device), c=args.c)
    

def retrofit_model(Ws):
    # Wrap all of the layers of the model
    for layer in range(num_layers):
        if type(layerlist[layer]) != LinearControlWrapper:
            layerlist[layer] = LinearControlWrapper(
                layerlist[layer],
                linear_probe=Ws[layer],
                p=args.p
            )
        else:
            layerlist[layer] = LinearControlWrapper(
                layerlist[layer].base_layer,
                linear_probe=Ws[layer],
                p=args.p
            )


retrofit_model(Ws)

results_dict = {}

for i, datum in tqdm(enumerate(data)):
    encoding = encode_data(tokenizer, 1, [datum], 1, model.config.max_position_embeddings, args.device)[0]

    results_dict[datum] = {}

    for layer in range(num_layers):
        retrofit_model(Ws)

        results_dict[data[i]][layer] =  {}

    if args.method == 'ours':
        for layer in range(num_layers):
            layerlist[layer].control_on()
    else:
        for layer in range(num_layers):
            layerlist[layer].control_off()

    for layer in range(num_layers):
        layerlist[layer].reset_logs()

    # Generate output
    print(data[i])

    try:
        outputs = model.generate(
            inputs=encoding['input_ids'], # batch size x seq len
            min_new_tokens=1,
            max_new_tokens=50,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
        )
    except IndexError:
        continue
    generated_tokens = outputs.sequences
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )

    # Surface forms log: the text generated + surprisal
    generated_tokens_text = [tokenizer.decode(token) for token in generated_tokens[0]]
    surprisals = [-float(score.cpu().numpy()) for score in transition_scores[0]]
    results_dict[data[i]]['generated_text'] = tokenizer.decode(*outputs[0], skip_special_tokens=True)
    results_dict[data[i]]['token'] = generated_tokens_text
    results_dict[data[i]]['surprisal'] = surprisals

    # Semantic scores log:
    for layer in range(num_layers):
        pre_adjust_scores = [float(score.item()) for score in layerlist[layer].pre_adjust_toxicity_log.copy()]
        post_adjust_scores = [float(score.item()) for score in layerlist[layer].post_adjust_toxicity_log.copy()]
        latency = layerlist[layer].latency.copy()
        results_dict[data[i]][layer]['pre_adjust_toxicity_prob'] = pre_adjust_scores
        results_dict[data[i]][layer]['post_adjust_toxicity_prob'] = post_adjust_scores
        results_dict[data[i]][layer]['inference_latency'] = latency

# Save data
if args.method == 'actadd':
    args.method = args.method + f'_c{args.c}_l{args.l}'
with open(f'/home/echeng/llm-control/experiments/{args.experiment}/control_results/{args.model_name.split("/")[-1]}_p_{args.p}_{args.method}.json', 'w') as f:
    json.dump(results_dict, f)
