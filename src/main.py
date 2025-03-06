import sys
import os
import argparse
import random
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Add the parent directory and src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Local imports
from config import YOUR_PATH, YOUR_TOKEN
from control_wrapper import LinearControlWrapper
from actadd_wrapper import ActAddWrapper
from instructions import *
from data_util import encode_data

# Set environment variables
os.environ['YOUR_PATH'] = os.getcwd()
ACCESS_TOKEN = YOUR_TOKEN

def parse_args():
    """Parse command line arguments for the training script."""
    parser = argparse.ArgumentParser(description='Training proof-of-concept for LLM control')
    
    # Model configuration
    parser.add_argument('--model_name', type=str, 
                       default="meta-llama/Meta-Llama-3-8B",
                       help='Name of the model to use')
    parser.add_argument('--method', type=str,
                       default='baseline',
                       choices=['baseline', 'ours'],
                       help='Training method to use')
    parser.add_argument('--layers', type=int, nargs='+',
                       help='List of layers to use for control')
    
    # Data configuration
    parser.add_argument('--dataset_name', type=str,
                       default=os.path.join(YOUR_PATH, 'elix_generations_gpt4omini_pref/train_shuffled_balanced.csv'),
                       help='Path to the dataset')
    parser.add_argument('--experiment', type=str,
                       default='elix',
                       help='Name of the experiment')
    
    # Training configuration
    parser.add_argument('--batch_size', type=int,
                       default=1,
                       help='Batch size for training')
    parser.add_argument('--device', type=str,
                       default='cuda',
                       help='Device to run on (cuda/cpu)')
    parser.add_argument('--random_seed', type=int,
                       help='Random seed for reproducibility')
    parser.add_argument('--continuous_tune', type=int,
                       default=0,
                       help='Whether to use continuous tuning')
    parser.add_argument('--p', type=float,
                       default=0.1,
                       help='Probability parameter for control')
    
    return parser.parse_args()

def setup_model_and_tokenizer(model_name, access_token):
    """Set up the model and tokenizer with appropriate configurations."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=access_token,
        load_in_8bit=True
    )
    
    # Configure tokenizer based on model type
    if 'Llama' in model_name:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    elif 'pythia' in model_name or 'mistral' in model_name:
        tokenizer.pad_token = tokenizer.eos_token
    elif 'OLMo' in model_name:
        model.config.max_position_embeddings = model.config.max_sequence_length
    
    model.eval()
    return model, tokenizer

def load_probes(model, args):
    """Load all linear probes for the model."""
    num_layers = model.config.num_hidden_layers
    Ws = []
    
    for layer in range(1, num_layers + 1):
        probe_path = os.path.join(
            YOUR_PATH,
            'experiments/elix/saved_probes',
            f'{args.model_name.split("/")[-1]}_linear_probe_layer_{layer}_rs42.pt'
        )
        
        print(f"Looking for probe at: {probe_path}")
        try:
            W = torch.load(probe_path).to(args.device)
            Ws.append(W)
            W.eval()
        except Exception as e:
            print(f"Error loading probe: {e}")
    
    return Ws

def retrofit_model(layerlist, Ws, args):
    """Wrap model layers with control wrappers."""
    num_layers = len(layerlist)
    for layer in range(num_layers):
        if not isinstance(layerlist[layer], LinearControlWrapper):
            layerlist[layer] = LinearControlWrapper(
                layerlist[layer],
                linear_probe=Ws[layer],
                p=args.p,
                continuous_tune=bool(args.continuous_tune)
            )
        else:
            layerlist[layer] = LinearControlWrapper(
                layerlist[layer].base_layer,
                linear_probe=Ws[layer],
                p=args.p,
                continuous_tune=bool(args.continuous_tune)
            )

def main():
    """Main execution function."""
    args = parse_args()
    random.seed(args.random_seed)
    
    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args.model_name, ACCESS_TOKEN)
    
    # Load probes
    Ws = load_probes(model, args)
    
    # Get layer list based on model type
    layerlist = model.gpt_neox.layers if 'pythia' in args.model_name else model.model.layers
    
    # Load dataset
    test_dataset = load_dataset("Asap7772/elix_generations_gpt4omini_pref")
    dataset = pd.DataFrame(test_dataset['test'])
    text_field = 'prompt'
    data = dataset[text_field].drop_duplicates().tolist()
    
    # Initialize results dictionary
    results_dict = {}
    
    # Process each data point
    for i, datum in tqdm(enumerate(data)):
        results_dict[datum] = {}
        #retrofit_model(layerlist, Ws, args)
        
        # Set up layer controls
        #for layer in range(len(layerlist)):
            #esults_dict[data[i]][layer] = {}
            #layerlist[layer].reset_logs()
            #if args.method == 'ours':
            #    layerlist[layer].control_on()
            #else:
             #   layerlist[layer].control_off()
        
        # Generate output
        print(datum)
        
        try:
            prompt = f"Question: {datum}\nAnswer:"
            encoding = encode_data(tokenizer, 1, [prompt], 1, 
                                  model.config.max_position_embeddings, args.device)[0]
            
            outputs = model.generate(
                inputs=encoding['input_ids'],
                min_new_tokens=1,
                max_new_tokens=400,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            # Process outputs
            generated_tokens = outputs.sequences
            transition_scores = model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            
            # Store results
            generated_tokens_text = [tokenizer.decode(token) for token in generated_tokens[0]]
            surprisals = [-float(score.cpu().numpy()) for score in transition_scores[0]]
            
            results_dict[datum].update({
                'generated_text': tokenizer.decode(*outputs[0], skip_special_tokens=True),
                'token': generated_tokens_text,
                'surprisal': surprisals
            })
            
            print(results_dict[datum]['generated_text'])
            
            # Store layer-specific scores
            #for layer in range(len(layerlist)):
            #    results_dict[data[i]][layer].update({
            #        'pre_adjust_toxicity_prob': [float(score.item()) for score in layerlist[layer].pre_adjust_toxicity_log.copy()],
            #        'post_adjust_toxicity_prob': [float(score.item()) for score in layerlist[layer].post_adjust_toxicity_log.copy()],
            #        'inference_latency': layerlist[layer].latency.copy()
            #    })
                
        except IndexError:
            continue
    
    # Save results
    results_dir = os.path.join(YOUR_PATH, 'experiments/elix/baseline_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save JSON results
    results_file = os.path.join(
        results_dir,
        f"{args.model_name.split('/')[-1]}_p_{args.p}_{args.method}.json"
    )
    with open(results_file, 'w') as f:
        json.dump(results_dict, f)
    
    # Save CSV results
    csv_data = [[datum, results_dict[datum]['generated_text']] for datum in data]
    df = pd.DataFrame(csv_data, columns=['Input', 'Generated Text'])
    df.to_csv(os.path.join(results_dir, 'generated_results.csv'), index=False)

if __name__ == '__main__':
    main()
