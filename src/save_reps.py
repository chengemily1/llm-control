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

parser = argparse.ArgumentParser(description='ID computation')

# Data selection
parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument('--dataset_name', type=str, default='/home/echeng/llm-control/stanford-nlp-imdb-sentiment')
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
elif 'OLMo' in args.model_name:
    model.config.max_position_embeddings = model.config.max_sequence_length

model.eval()


# Load and shuffle the dataset
def balance_labels(label_name, df):
    labels = df[label_name]
    minority = 1 if sum(labels) < 0.5 * len(labels) else 0
    
    # get all the minority labels
    minority_df = df[df[label_name] == minority]
    majority_df = df[df[label_name] == (1 if not minority else 0)].sample(len(minority_df))
    
    return pd.concat([minority_df, majority_df])

if not os.path.exists(args.dataset_name + '/train_shuffled_balanced.csv'):
    dataset = pd.read_csv(args.dataset_name + '/train_shuffled.csv')
    balanced_df = balance_labels('toxic', dataset).sample(frac=1)
    balanced_df.to_csv(args.dataset_name + '/train_shuffled_balanced.csv')
else:
    balanced_df = pd.read_csv(args.dataset_name + '/train_shuffled_balanced.csv')

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

dataset = pd.read_csv(args.dataset_name + '/train_shuffled_balanced.csv')
data = list(dataset['text'])

# tokenize data
encodings = encode_data(tokenizer, len(data), data, args.batch_size, model.config.max_position_embeddings, args.device)

def last_token_rep(x, attention_mask, padding='right'):
    seq_len = attention_mask.sum(dim=1)
    indices = (seq_len - 1)
    last_token_rep = x[torch.arange(x.size(0)), indices] if padding=='right' else x[torch.arange(x.size(0)), -1]
    return last_token_rep.cpu()

# PROCESS AND SAVE REPS
with torch.no_grad():
    representations = []
    for batch in tqdm(encodings):
        output = model(batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)['hidden_states']
        pooled_output = tuple([last_token_rep(layer, batch['attention_mask'], padding=tokenizer.padding_side) for layer in output])
        representations.append(pooled_output)
    representations = [list(batch) for batch in zip(*representations)]
    representations = [torch.cat(batches, dim=0) for batches in representations]
    print('Layer 1 reps shape: ')
    print(representations[1].shape)
    torch.save(representations, f'/home/echeng/llm-control/experiments/{args.experiment}/saved_reps/{args.model_name.split("/")[-1]}_reps.pt')
