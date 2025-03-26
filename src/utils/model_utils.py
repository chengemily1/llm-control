import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pdb

def setup_model_and_tokenizer(model_name, access_token, device):
    """Set up and configure the model and tokenizer."""
    # Configure tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    
    # Ensure pad token is set before model loading
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    
    # Configure model with explicit pad token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=access_token,
        load_in_8bit=True,
        pad_token_id=tokenizer.pad_token_id,  # Set pad_token_id to match tokenizer
    )

    # Additional model-specific configurations
    if 'OLMo' in model_name:
        model.config.max_position_embeddings = model.config.max_sequence_length
    
    # Ensure model config matches tokenizer settings
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.padding_side = tokenizer.padding_side

    model.eval()
    return model, tokenizer

def load_probes(model, args, your_path):
    """Load all linear probes for the model."""
    num_layers = model.config.num_hidden_layers
    Ws = []
    
    for layer in range(1, num_layers + 1):
        probe_path = os.path.join(
            your_path,
            'experiments/gms8k/saved_probes',
            f'{args.model_name.split("/")[-1]}_layer_{layer}_rs42_probe.pt'
        ) 
        
        print(f"Looking for probe at: {probe_path}")
        try:
            W = torch.load(probe_path) 
            W.eval()
            Ws.append(W)
        except Exception as e:
            print(f"Error loading probe: {e}")
    
    return Ws

def get_layer_list(model, model_name):
    """Get the appropriate layer list based on model type."""
    return model.gpt_neox.layers if 'pythia' in model_name else model.model.layers 

def get_layers_by_r2(model_name: str, base_path: str, r2_threshold: float = 0.5) -> list:
    """Get layers whose probes achieved R² scores above the threshold.
    
    Args:
        model_name (str): Name of the model
        base_path (str): Base path to experiments directory
        r2_threshold (float): Minimum R² score to consider a layer (default: 0.5)
    
    Returns:
        list: List of layer indices with R² scores above threshold
    """
    import os
    import json
    import glob
    
    # Get the model name without path
    model_short_name = model_name.split('/')[-1]
    
    # Find all probe result files for this model
    results_path = os.path.join(base_path, 'experiments/gms8k/probing_results')
    pattern = os.path.join(results_path, f"{model_short_name}_layer_*_validation_results_over_training.json")
    result_files = glob.glob(pattern)
    
    good_layers = []
    for file_path in result_files:
        # Extract layer number from filename
        layer_str = file_path.split('layer_')[-1].split('_')[0]
        layer_num = int(layer_str)
        
        # Load results
        with open(file_path, 'r') as f:
            results = json.load(f)
        
        # Get final R² score
        final_r2 = results['val_r2'][-1]
        
        if final_r2 >= r2_threshold:
            good_layers.append(layer_num)
    
    # Sort layers for consistency
    good_layers.sort()
    
    if not good_layers:
        print(f"Warning: No layers found with R² score above {r2_threshold}")
    else:
        print(f"Found {len(good_layers)} layers with R² score above {r2_threshold}: {good_layers}")
    
    return good_layers 