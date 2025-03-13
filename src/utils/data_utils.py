import torch
from datasets import load_dataset
import pandas as pd
import os
import json

def encode_data(tokenizer, N, data, batch_size, max_length, device):
    """Encode data for model input.
    
    Args:
        tokenizer: The tokenizer to use
        N (int): Number of times to process the data
        data (list): List of strings to encode
        batch_size (int): Batch size for processing
        max_length (int): Maximum sequence length
        device (str): Device to put tensors on
    
    Returns:
        list: List of encoded batches with input_ids and attention_mask
    """
    # If the input data is text
    if isinstance(data[0], str):
        encodings = []
        for i in range(0, N, batch_size):
            tokenizer_output = tokenizer(
                data[i:i + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                return_attention_mask=True  # Explicitly request attention mask
            )
            tokenizer_output['input_ids'] = tokenizer_output['input_ids'].to(device)
            tokenizer_output['attention_mask'] = tokenizer_output['attention_mask'].to(device)
            encodings.append(tokenizer_output)
    else:  # input data is tokens
        max_len = max(len(sentence) for sentence in data)
        data = [sentence for sentence in data if len(sentence) > 2]
        
        # Create input_ids
        encodings = []
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size]
            input_ids = torch.stack([
                torch.tensor(sentence[1:], device=device) for sentence in batch_data
            ]).squeeze(1)
            
            # Create attention mask (1 for tokens, 0 for padding)
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            
            # Pad sequences to max_len
            if input_ids.size(1) < max_len:
                padding = torch.full(
                    (input_ids.size(0), max_len - input_ids.size(1)),
                    tokenizer.pad_token_id,
                    device=device
                )
                input_ids = torch.cat([input_ids, padding], dim=1)
                attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros_like(padding)
                ], dim=1)
            
            encodings.append({
                'input_ids': input_ids.to(device),
                'attention_mask': attention_mask.to(device)
            })
    
    return encodings

def load_and_process_dataset(experiment='reviews'):
    """Load and preprocess the dataset.
    
    Args:
        experiment (str): Type of experiment ('elix' or 'reviews')
    """
    if experiment == 'elix':
        test_dataset = load_dataset("Asap7772/elix_generations_gpt4omini_pref")
    elif experiment == 'reviews':
        test_dataset = load_dataset("Asap7772/steered_reviews_full_autolabel_gpt4o_pref")
    dataset = pd.DataFrame(test_dataset['test'])
    text_field = 'prompt'

    # Drop duplicates and format as question-answer pairs
    dataset = dataset.drop_duplicates(subset=[text_field])
    data = [f"Question: {text}\nAnswer: " for text in dataset[text_field]]
    
    return data

def process_generation_output(tokenizer, outputs, transition_scores):
    """Process model generation outputs."""
    generated_tokens = outputs.sequences
    generated_tokens_text = [tokenizer.decode(token) for token in generated_tokens[0]]
    surprisals = [-float(score.cpu().numpy()) for score in transition_scores[0]]
    
    return {
        'generated_text': tokenizer.decode(*outputs[0], skip_special_tokens=True),
        'token': generated_tokens_text,
        'surprisal': surprisals
    }

def save_results(results_dict, output_path):
    """Save results to a JSON file, creating directories if needed."""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving results to: {output_path}")
    
    with open(output_path, 'w') as f:
        json.dump(results_dict, f) 