"""Script to train classification probes for analyzing model representations."""

import os
import sys
import random
import argparse
import torch
import pandas as pd

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.config_utils import load_config
from src.utils.probe_utils_classification import (
    ClassificationProbe, 
    get_train_val_split_balanced,
    train_classification_probe, 
    save_classification_probe_results
)
from src.probe_training.experiment_config import ReviewsExperimentConfig

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train classification probe')
    # Required arguments
    parser.add_argument('--layer', type=int, required=True,
                       help='Layer to probe')
    
    # Optional arguments that override experiment config
    parser.add_argument('--model_name', type=str,
                       help='Override default model name')
    parser.add_argument('--device', type=str,
                       help='Override default device')
    parser.add_argument('--batch_size', type=int,
                       help='Override default batch size')
    parser.add_argument('--num_epochs', type=int,
                       help='Override default number of epochs')
    parser.add_argument('--learning_rate', type=float,
                       help='Override default learning rate')
    parser.add_argument('--random_seed', type=int,
                       help='Override default random seed')
    parser.add_argument('--downsample', type=float,
                       help='Override default downsample ratio')
    parser.add_argument('--reps_suffix', type=str, choices=['final', '5000'],
                       help='Which representation file to use (final or 5000)')
    
    # Other arguments
    parser.add_argument('--save', type=int, choices=[0,1], default=1,
                       help='Whether to save the probe')
    
    # Get the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    default_config = os.path.join(project_root, 'src', 'config.json')
    parser.add_argument('--config', default=default_config,
                       help='Path to config file')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Create experiment config
    exp_config = ReviewsExperimentConfig()
    
    # Update config with any provided args
    for key, value in vars(args).items():
        if value is not None and hasattr(exp_config, key):
            setattr(exp_config, key, value)
    
    # Set random seed
    random.seed(exp_config.random_seed)
    torch.manual_seed(exp_config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(exp_config.random_seed)
    
    # Load project config
    config = load_config(args.config)
    
    # Get paths
    paths = exp_config.get_paths(config['base_path'])
    
    # Setup device
    device = 'cuda:0' if exp_config.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    
    # Load dataset
    dataset = pd.read_csv(paths['data']['processed'])
    
    # Load representations
    try:
        representations = torch.load(paths['representations']['layer'](args.layer))
        # If representations is a dict, try to get the layer
        if isinstance(representations, dict):
            representations = representations[args.layer]
        # If it's a tensor, use it directly
        elif isinstance(representations, torch.Tensor):
            representations = representations
        # If it's a list of tensors, get the specified layer
        elif isinstance(representations, list):
            if args.layer < len(representations):
                representations = representations[args.layer]
            else:
                raise ValueError(f"Layer {args.layer} is out of range. Max layer is {len(representations)-1}")
        else:
            raise ValueError(f"Unexpected representation format: {type(representations)}")
    except Exception as e:
        print(f"Error loading representations: {e}")
        return
    
    # Get labels
    labels = list(dataset[exp_config.label_field])[:len(representations)]
    
    # Split data
    train_features, train_labels, val_features, val_labels = get_train_val_split_balanced(
        representations, labels, device=device
    )
    
    # Downsample if needed
    if exp_config.downsample < 1:
        train_size = int(len(train_features) * exp_config.downsample)
        train_features = train_features[:train_size]
        train_labels = train_labels[:train_size]
    
    # Initialize model
    input_dim = representations[0].shape[-1]
    model = ClassificationProbe(input_dim=input_dim).to(device)
    
    # Train model
    model, metrics = train_classification_probe(
        model, train_features, train_labels, val_features, val_labels,
        num_epochs=exp_config.num_epochs,
        learning_rate=exp_config.learning_rate,
        device=device
    )
    
    # Save results if requested
    if args.save:
        model_name = exp_config.model_name.split('/')[-1]
        layer_str = f"layer_{args.layer}"
        rs_str = f"rs{args.random_seed}"
        ds_str = f"_downsample_{args.downsample}" if (args.downsample and args.downsample < 1) else ""
        save_name = f"{model_name}_{layer_str}_{rs_str}{ds_str}"

        save_classification_probe_results(
            metrics=metrics,
            model=model,
            save_name=save_name,
            probes_dir=paths['probes']['base'],
            results_dir=paths['probes']['results'],
            save=args.save
        )

if __name__ == "__main__":
    main() 