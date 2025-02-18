import argparse
import torch
from transformers import AutoModelForCausalLM, GPTNeoXTokenizerFast, AutoTokenizer
from datasets import load_dataset
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import pdb
from config import YOUR_PATH, YOUR_TOKEN

parser = argparse.ArgumentParser(description='ID computation')

# Data selection
parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3-8B")
parser.add_argument('--dataset_name', type=str, default=f'{YOUR_PATH}/OpenThoughts-114k-math')
parser.add_argument('--attn', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--experiment', default='sentiment', 
                   choices=['sentiment', 'toxicity', 'formality', 'reasoning'])
args = parser.parse_args()
print(args)

ACCESS_TOKEN= YOUR_TOKEN

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

# Shuffle the dataset
text_col = {'toxicity': 'comment_text', 'sentiment': 'text', 'formality': 'sentence', 'reasoning': 'problem_solution'}

def balance_labels(label_name, df):
    labels = df[label_name]
    minority = 1 if labels.sum() < 0.5 * len(labels) else 0
    
    # get all the minority labels
    minority_df = df[df[label_name] == minority]
    majority_df = df[df[label_name] == (1 if not minority else 0)].sample(len(minority_df))
    
    return pd.concat([minority_df, majority_df])

os.makedirs(args.dataset_name, exist_ok=True)
if not os.path.exists(args.dataset_name + '/train_shuffled_balanced.csv'):
    # For handling HuggingFace dataset
    if args.experiment == 'reasoning':
        # Load dataset based on experiment type
        dataset = load_dataset("open-r1/OpenThoughts-114k-math")
        print("Dataset loaded successfully!")
        
        # Convert to DataFrame
        df = pd.DataFrame(dataset['train'])
        # Drop rows with NaN in either 'problem' or 'solution'
        df = df.dropna(subset=['problem', 'solution'])
        
        # Create the concatenated column
        df['problem_solution'] = df['problem'] + ' ' + df['solution']
        
        # Balance the dataset and shuffle it
        balanced_df = balance_labels('correct', df).sample(frac=1)
        balanced_df.to_csv(args.dataset_name + '/train_shuffled_balanced.csv', index=False)
    else:
    # For other experiments using local files
        dataset = pd.read_csv(args.dataset_name + '/train_shuffled.csv')
        balanced_df = balance_labels('toxic', dataset).sample(frac=1)
        balanced_df.to_csv(args.dataset_name + '/train_shuffled_balanced.csv')

    # Drop rows where the target text column is NaN        
    balanced_df = balanced_df.dropna(subset=[text_col[args.experiment]])
    balanced_df.to_csv(args.dataset_name + '/train_shuffled_balanced.csv')

def encode_data(tokenizer, N, data, batch_size, max_length, device):
    # last_k (int): only use the last k tokens of the input

    # If the input data is text
    if type(data[0]) == str:
        
        encodings = []
        for i in tqdm(range(0, N, batch_size)):
            tokenizer_output = tokenizer(data[i: i + batch_size], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            tokenizer_output['input_ids'] = tokenizer_output['input_ids'].to(device)
            tokenizer_output['attention_mask'] = tokenizer_output['attention_mask'].to(device)
            encodings.append(tokenizer_output)
        
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

dataset = pd.read_csv(args.dataset_name + '/train_shuffled_balanced.csv', keep_default_na=False)
data = list(dataset[text_col[args.experiment]])

# tokenize data
if not args.attn:
    print("Encoding data")
    encodings = encode_data(tokenizer, len(data), data, args.batch_size, model.config.max_position_embeddings, args.device)
    print("done")
    def last_token_rep(x, attention_mask, padding='right'):
        seq_len = attention_mask.sum(dim=1)
        indices = (seq_len - 1)
        last_token_rep = x[torch.arange(x.size(0)), indices] if padding=='right' else x[torch.arange(x.size(0)), -1]
        return last_token_rep.cpu()

    # PROCESS AND SAVE REPS
    print("processing and saving reps")
    folder = f"{YOUR_PATH}/experiments/{args.experiment}/saved_reps/"
    os.makedirs(folder, exist_ok=True)
    with torch.no_grad():
        representations = []
        for i, batch in tqdm(enumerate(encodings)):
            output = model(batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)['hidden_states']
            pooled_output = tuple([last_token_rep(layer, batch['attention_mask'], padding=tokenizer.padding_side) for layer in output])
            representations.append(pooled_output)
            del output 
            torch.cuda.empty_cache()

            if (i + 1) % 5000 == 0:
                print(f"Saving representations...")
                representations = [torch.cat(batches, dim=0) for batches in zip(*representations)]
                torch.save(representations, os.path.join(folder, f"{args.model_name.split('/')[-1]}_reps_part_{i+1}.pt"))
                del representations  # Free memory
                representations = []  # Reset list
                torch.cuda.empty_cache()
      
        if representations:
            representations = [torch.cat(batches, dim=0) for batches in zip(*representations)]
            torch.save(representations, os.path.join(folder, f"{args.model_name.split('/')[-1]}_reps_final.pt"))
else:
    encodings = []
    for datum in data:
        encodings.append(encode_data(tokenizer, 1, [datum], 1, model.config.max_position_embeddings, args.device)[0])

    if 'Llama' in args.model_name or 'mistral' in args.model_name:
        HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]
    elif 'pythia' in args.model_name:
        HEADS = [f"gpt_neox.layers.{i}.attention.dense" for i in range(model.config.num_hidden_layers)]

    head_embeds = []
    with torch.no_grad():
        for i, prompt in tqdm(enumerate(encodings)):
            prompt = prompt['input_ids']   
            with TraceDict(model, HEADS) as ret:
                output = model(prompt)
            head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]

            head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze().numpy()
            if len(tuple(head_wise_hidden_states.shape)) == 3:
                head_wise_hidden_states = head_wise_hidden_states[:, -1, :]
            attn_dim = head_wise_hidden_states.shape[-1] // model.config.num_attention_heads
            head_wise_hidden_states = head_wise_hidden_states.reshape((head_wise_hidden_states.shape[0], model.config.num_attention_heads, attn_dim))
            head_embeds.append(head_wise_hidden_states)

    head_embeds = np.array(head_embeds)
    folder = f"{YOUR_PATH}/llm-control/experiments/{args.experiment}/saved_attn_reps/"
    os.makedirs(folder, exist_ok=True)

    np.save(os.path.join(folder, '{args.model_name.split("/")[-1]}_reps.npy'), head_embeds)
