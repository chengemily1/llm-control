import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Define YOUR_PATH as the current directory
YOUR_PATH = '/scr/biggest/carmen/llm-control'
# Define YOUR_TOKEN with your Hugging Face token
YOUR_TOKEN = "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"

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
from src.data_util import encode_data

# Set the HF_HOME environment variable to your current working directory
os.environ['YOUR_PATH'] = os.getcwd()
ACCESS_TOKEN = YOUR_TOKEN 

parser = argparse.ArgumentParser(description='training proof-of-concept')

# Data selection
parser.add_argument('--dataset_name', type=str, default='../../shared_data/llm_control/train_shuffled_balanced.csv')
parser.add_argument('--experiment', type=str, default='reasoning')
parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3-8B")
parser.add_argument('--method', default='ours', choices=['baseline', 'ours', 'actadd', 'instruct', 'fudge'])
parser.add_argument('--layers', metavar='N', type=int, nargs='+',
                        help='an integer or a list of integers')
parser.add_argument('--continuous_tune', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--p', type=float, default=0.1)
#parser.add_argument('--c', help="Actadd intervention strength", type=float, default=3)
parser.add_argument('--random_seed', type=int)
#parser.add_argument('--l', default=6, type=int)
#parser.add_argument('--s', default=None, type=float)
args = parser.parse_args()

#exp = 'sentiment' if args.experiment in ('formality', 'sentiment') else 'toxicity' # reuse the sentiment prompts for formlaity
#args.dataset_name = os.path.join(YOUR_PATH, 'experiments', f'test_{exp}.csv') # last minute switch
   
random.seed(args.random_seed)
# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=ACCESS_TOKEN)
model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                             token=ACCESS_TOKEN,
                                             #load_in_8bit=True
                                            ).cuda()

if 'Llama' in args.model_name:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
elif 'pythia' in args.model_name or 'mistral' in args.model_name:
    tokenizer.pad_token = tokenizer.eos_token
elif 'OLMo' in args.model_name:
    model.config.max_position_embeddings = model.config.max_sequence_length

model.eval()
# pdb.set_trace()

# Load the dataset
dataset = pd.read_csv(args.dataset_name) 
text_field = 'problem'
data = list(dataset[text_field])
# Load the reps used for training
model_reps_path = os.path.join('../../shared_data/llm_control/', 'Meta-Llama-3-8B_reps_final_new.pt')
model_reps = torch.load(model_reps_path)
# Exclude the first len(model_reps) rows from the dataset
data = data[len(model_reps):len(model_reps)+10]  # Update to skip the first len(model_reps) rows

# ORIGINAL BASELINE
model.eval()
print(args)
# pdb.set_trace()

if 'pythia' in args.model_name:
    layerlist = model.gpt_neox.layers
else:
    layerlist = model.model.layers

results_dict = {}

for i, datum in tqdm(enumerate(data)):
    encoding = encode_data(tokenizer, 1, [datum], 1, model.config.max_position_embeddings, args.device)[0]

    results_dict[datum] = {}
    
    outputs = model.generate(
        inputs=encoding['input_ids'], # batch size x seq len
        attention_mask=encoding['attention_mask'],
        min_new_tokens=1,
        max_new_tokens=200,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True
    )
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

    print(results_dict[data[i]]['generated_text'])

# Define the path for the results file
results_dir = '../../shared_data/llm_control/experiments/control_results/'
os.makedirs(results_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Now you can safely open the file for writing
with open(os.path.join(results_dir, f"{args.model_name.split('/')[-1]}_p_{args.p}_{args.method}.json"), 'w') as f:
    json.dump(results_dict, f)

# Define the path for the results CSV file
results_csv_path = os.path.join(results_dir, 'generated_results.csv')

# Prepare data for the CSV
csv_data = []
for i in range(len(data)):
    csv_data.append([data[i], results_dict[data[i]]['generated_text']])

# Create a DataFrame and save it to a CSV file
df = pd.DataFrame(csv_data, columns=['Input', 'Generated Text'])
df.to_csv(results_csv_path, index=False)  # Save to CSV without the index
