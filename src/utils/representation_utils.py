"""Utilities for extracting and saving model representations."""

import torch
import os
from tqdm import tqdm
import numpy as np

def save_layer_representations(model, tokenizer, data, batch_size, device, output_folder, model_name):
    """Extract and save layer representations."""
    print("Processing and saving representations")
    os.makedirs(output_folder, exist_ok=True)
    
    with torch.no_grad():
        representations = []
        for i, batch in tqdm(enumerate(data)):
            output = model(batch['input_ids'], 
                         attention_mask=batch['attention_mask'], 
                         output_hidden_states=True)['hidden_states']
            
            pooled_output = tuple([last_token_rep(layer, batch['attention_mask'], 
                                                padding=tokenizer.padding_side) for layer in output])
            representations.append(pooled_output)
            del output 
            torch.cuda.empty_cache()

            if (i + 1) % 5000 == 0:
                save_batch_representations(representations, output_folder, model_name, i+1)
                representations = []
                torch.cuda.empty_cache()
      
        if representations:
            save_batch_representations(representations, output_folder, model_name, "final")

def last_token_rep(x, attention_mask, padding='right'):
    """Get the representation of the last token."""
    seq_len = attention_mask.sum(dim=1)
    indices = (seq_len - 1)
    last_token_rep = x[torch.arange(x.size(0)), indices] if padding=='right' else x[torch.arange(x.size(0)), -1]
    return last_token_rep.cpu()

def save_batch_representations(representations, folder, model_name, batch_id):
    """Save a batch of representations to disk, one file per layer."""
    print(f"Saving representations batch {batch_id}...")
    # Transpose the list of tuples into a list of lists, where each inner list contains all representations for one layer
    layer_representations = list(zip(*representations))
    
    # For each layer, concatenate its representations and save to a separate file
    for layer_idx, layer_reps in enumerate(layer_representations):
        layer_tensor = torch.cat(layer_reps, dim=0)
        layer_file = os.path.join(folder, f"{model_name.split('/')[-1]}_layer_{layer_idx}_reps.pt")
        torch.save(layer_tensor, layer_file)
        del layer_tensor
        torch.cuda.empty_cache()
    
    del layer_representations
    torch.cuda.empty_cache()

def save_attention_representations(model, tokenizer, data, output_folder, model_name, device):
    """Extract and save attention-based representations."""
    if 'Llama' in model_name or 'mistral' in model_name:
        heads = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]
    elif 'pythia' in model_name:
        heads = [f"gpt_neox.layers.{i}.attention.dense" for i in range(model.config.num_hidden_layers)]
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    head_embeds = []
    with torch.no_grad():
        for i, prompt in tqdm(enumerate(data)):
            prompt = prompt['input_ids']   
            with TraceDict(model, heads) as ret:
                output = model(prompt)
            head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in heads]

            head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim=0).squeeze().numpy()
            if len(tuple(head_wise_hidden_states.shape)) == 3:
                head_wise_hidden_states = head_wise_hidden_states[:, -1, :]
            
            attn_dim = head_wise_hidden_states.shape[-1] // model.config.num_attention_heads
            head_wise_hidden_states = head_wise_hidden_states.reshape(
                (head_wise_hidden_states.shape[0], model.config.num_attention_heads, attn_dim)
            )
            head_embeds.append(head_wise_hidden_states)

    head_embeds = np.array(head_embeds)
    os.makedirs(output_folder, exist_ok=True)
    np.save(os.path.join(output_folder, f'{model_name.split("/")[-1]}_reps.npy'), head_embeds) 