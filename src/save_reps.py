#########################
# Imports
#########################
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

# Set the HF_HOME environment variable to your current working directory
os.environ['YOUR_PATH'] = os.getcwd()

parser = argparse.ArgumentParser(description='ID computation')

#########################
# Arguments
#########################
parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3-8B")
parser.add_argument('--dataset_name', type=str, default=f'{YOUR_PATH}/elix_generations_gpt4omini_pref')
parser.add_argument('--attn', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--experiment', default='elix')
parser.add_argument('--user', default='child', choices=['child', 'preteen', 'teenager', ' young adult', 'expert'])
args = parser.parse_args()
print(args)

text_col = {'elix': 'prompt_response'}

ACCESS_TOKEN= YOUR_TOKEN

#########################
# Load the model and tokenizer
#########################   
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

#########################
# Load the dataset
#########################
os.makedirs(args.dataset_name, exist_ok=True)
if not os.path.exists(args.dataset_name + '/train_shuffled_balanced.csv'):
    ############## ELIX DATASET ##############
    if args.experiment == 'elix':
        # Define the users
        user_id = {'child': 1, 'preteen': 2, 'teenager': 3, ' young adult': 4, 'expert': 5}
        # Load dataset
        dataset = load_dataset("Asap7772/elix_generations_gpt4omini_pref")
        print("Dataset loaded successfully!")

        # Convert to DataFrame
        df = pd.DataFrame(dataset['train'])
        
        # Remove unwanted columns
        columns_to_drop = ['level_x', 'level_y', 'model_name_x', 'model_name_y', 'scorer_level', '__index_level_0__']
        df = df.drop(columns=columns_to_drop)
        
        # Filter by scorer level
        df = df[df['scorer_level_id'] == user_id[args.user]]
        df = df.drop(columns='scorer_level_id')
        
        # Create scoring system based on preferences
        def calculate_level_scores(df):
            # Initialize scores for each level
            level_scores = {}
            
            # Count wins for each level
            for _, row in df.iterrows():
                if row['label'] == 0:  # response_x preferred
                    level_scores[row['level_id_x']] = level_scores.get(row['level_id_x'], 0) + 1
                else:  # response_y preferred
                    level_scores[row['level_id_y']] = level_scores.get(row['level_id_y'], 0) + 1
            
            # Normalize scores by number of comparisons each level participated in
            level_counts = {}
            for _, row in df.iterrows():
                level_counts[row['level_id_x']] = level_counts.get(row['level_id_x'], 0) + 1
                level_counts[row['level_id_y']] = level_counts.get(row['level_id_y'], 0) + 1
            
            normalized_scores = {
                level: score / level_counts[level] 
                for level, score in level_scores.items()
            }
            
            return normalized_scores
        
        # Calculate scores
        level_scores = calculate_level_scores(df)
        
        print("\nLevel scores (normalized by number of comparisons):")
        for level, score in sorted(level_scores.items()):
            print(f"Level {level}: {score:.3f}")
        
        # Reorganize dataset into separate rows for each response
        def split_comparisons(df):
            # Create DataFrame for response_x
            df_x = df.copy()
            df_x['response'] = df_x['response_x']
            df_x['level_id'] = df_x['level_id_x']
            df_x['score'] = df_x['level_id_x'].map(level_scores)
            df_x['prompt_response'] = df_x['prompt'] + ' ' + df_x['response_x']
            
            # Create DataFrame for response_y
            df_y = df.copy()
            df_y['response'] = df_y['response_y']
            df_y['level_id'] = df_y['level_id_y']
            df_y['score'] = df_y['level_id_y'].map(level_scores)
            df_y['prompt_response'] = df_y['prompt'] + ' ' + df_y['response_y']
            
            # Combine and clean up
            combined_df = pd.concat([df_x, df_y], ignore_index=True)
            
            # Keep only relevant columns
            columns_to_keep = ['prompt', 'response', 'prompt_response', 'level_id', 'score']
            combined_df = combined_df[columns_to_keep]
            
            # Remove duplicates (in case same response appears in multiple comparisons)
            combined_df = combined_df.drop_duplicates(subset=['prompt_response'])
            
            return combined_df
        
        def balance_by_score(df):
            # Get the minimum count of any score
            score_counts = df.groupby('score').size()
            min_count = score_counts.min()
            
            # For each score, sample down to the minimum count
            balanced_dfs = []
            for score in df['score'].unique():
                score_df = df[df['score'] == score]
                if len(score_df) > min_count:
                    # Randomly sample down to min_count
                    balanced_df = score_df.sample(n=min_count, random_state=42)
                else:
                    balanced_df = score_df
                balanced_dfs.append(balanced_df)
            
            # Combine all balanced dataframes
            balanced_df = pd.concat(balanced_dfs, ignore_index=True)
            
            # Print statistics about the balancing
            print("\nBefore balancing:")
            print(df.groupby('score').size())
            print("\nAfter balancing:")
            print(balanced_df.groupby('score').size())
            
            return balanced_df
        
        # First reorganize the dataset
        reorganized_df = split_comparisons(df)
        
        # Then balance by score
        reorganized_df = balance_by_score(reorganized_df)
        
        print(f"\nOriginal dataset shape: {df.shape}")
        print(f"Reorganized dataset shape: {reorganized_df.shape}")
        print("\nReorganized dataset columns:")
        print(reorganized_df.columns.tolist())
        
        # Count rows for each score
        print("\nNumber of rows for each score:")
        score_counts = reorganized_df.groupby('score').size().sort_index()
        for score, count in score_counts.items():
            print(f"Score {score:.3f}: {count} rows")
        
        # Shuffle and save the reorganized dataset
        reorganized_df = reorganized_df.sample(frac=1, random_state=42).reset_index(drop=True)
        reorganized_df.to_csv(args.dataset_name + '/train_shuffled_balanced.csv', index=False)
        print(f"\nReorganized dataset saved to {args.dataset_name}/train_shuffled_balanced.csv")

#########################
# ENCODE DATA
#########################
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


if not args.attn:
    print("Encoding data")
    encodings = encode_data(tokenizer, len(data), data, args.batch_size, model.config.max_position_embeddings, args.device)
    print("done")
    def last_token_rep(x, attention_mask, padding='right'):
        seq_len = attention_mask.sum(dim=1)
        indices = (seq_len - 1)
        last_token_rep = x[torch.arange(x.size(0)), indices] if padding=='right' else x[torch.arange(x.size(0)), -1]
        return last_token_rep.cpu()

    #########################
    # PROCESS AND SAVE REPS
    #########################   
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

#########################
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
