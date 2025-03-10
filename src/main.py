"""Main script for training and evaluation."""

import sys
import os
import argparse
import torch
import random
from tqdm import tqdm

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.utils.data_utils import (
    encode_data, 
    load_and_process_dataset, 
    process_generation_output, 
    save_results
)
from src.utils.config_utils import load_config
from src.utils.model_utils import (
    setup_model_and_tokenizer,
    load_probes,
    get_layer_list,
    get_layers_by_r2
)
from src.steering_methods.model_wrapper import (
    retrofit_model,
    collect_layer_metrics,
    print_intervention_summary
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='training proof-of-concept')
    parser.add_argument('--experiment', type=str, default='elix')
    parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument('--method', default='liseco', choices=['baseline', 'liseco', 'instruct'])
    parser.add_argument('--layers', metavar='N', type=int, nargs='+', 
                       help='an integer or a list of integers (if not provided, will use layers with R² > 0.5)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--liseco_lower', type=float, default=0.7)
    parser.add_argument('--liseco_upper', type=float, default=1)
    parser.add_argument('--liseco_map', default='identity', choices=['sigmoid', 'tanh', 'identity'])
    parser.add_argument('--s', default=None, type=float)
    parser.add_argument('--r2_threshold', type=float, default=0,
                       help='R² threshold for automatic layer selection')
    parser.add_argument('--config', default='/scratch/llm-control/src/config.json', help='path to config file')
    return parser.parse_args()

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args.model_name, config['access_token'], args.device)
    
    # Get layers to control
    if args.layers is None:
        print("\nNo layers specified, selecting layers based on R² scores...")
        selected_layers = get_layers_by_r2(args.model_name, config['base_path'], args.r2_threshold)
        if not selected_layers:
            print("No layers found with sufficient R² scores. Using all layers.")
            selected_layers = list(range(model.config.num_hidden_layers))
    else:
        selected_layers = args.layers
        print(f"\nUsing manually specified layers: {selected_layers}")
    
    # Print intervention summary
    liseco_params = {
        'liseco_lower': args.liseco_lower,
        'liseco_upper': args.liseco_upper,
        'liseco_map': args.liseco_map,
        's': args.s
    } if args.method == 'liseco' else None
    
    print_intervention_summary(
        model_name=args.model_name,
        method=args.method,
        selected_layers=selected_layers,
        liseco_params=liseco_params
    )
    
    # Load probes and get layer list
    Ws = load_probes(model, args, config['base_path'])

    layerlist = get_layer_list(model, args.model_name)
    
    # Load and process dataset
    data = load_and_process_dataset()
    
    # Initialize results dictionary
    results_dict = {}
    
    # Process each prompt
    for i, datum in tqdm(enumerate(data)):
        # Encode input
        encoding = encode_data(tokenizer, 1, [datum], 1, model.config.max_position_embeddings, args.device)[0]
        
        # Initialize results for this prompt
        results_dict[datum] = {}
        
        # Setup model layers
        retrofit_model(model, layerlist, Ws, args)
        
        # Initialize layer-specific results
        num_layers = model.config.num_hidden_layers
        for layer in range(num_layers):
            results_dict[data[i]][layer] = {}
        
        # Configure control based on method
        selected_layers_set = set(selected_layers)
        for layer in range(num_layers):
            if layer in selected_layers_set and args.method == 'liseco':
                layerlist[layer].control_on()
            else:
                layerlist[layer].control_off()
            layerlist[layer].reset_logs()
        
        # Generate output
        print(f"\nProcessing prompt: {data[i]}")
        try:
            outputs = model.generate(
                inputs=encoding['input_ids'],
                min_new_tokens=1,
                max_new_tokens=100,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                repetition_penalty=1.0,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Process generation output
            transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
            generation_results = process_generation_output(tokenizer, outputs, transition_scores)
            results_dict[data[i]].update(generation_results)
            
            print(f"Generated text: {generation_results['generated_text']}")
            
            # Collect metrics for each layer
            for layer in range(num_layers):
                results_dict[data[i]][layer].update(collect_layer_metrics(layerlist, layer))
                
        except IndexError:
            print(f"Skipping prompt due to IndexError: {datum}")
            continue
    
    # Save results
    output_path = f'{config["base_path"]}/experiments/{args.experiment}/control_results/{args.model_name.split("/")[-1]}_{args.method}.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_results(results_dict, output_path)

if __name__ == "__main__":
    main()
