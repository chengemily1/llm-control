"""Script to train probes for analyzing model representations."""

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
from src.utils.probe_utils import LinearProbe, get_train_val_split, train_probe, save_probe_results
from src.probe_training.experiment_config import ExperimentConfig

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train probe')
    # Required arguments
    parser.add_argument('--layer', type=int, required=True,
                       help='Layer to probe')
    
    # Optional arguments that override experiment config
    parser.add_argument('--model_name', type=str,
                       help='Override default model name')
    parser.add_argument('--experiment', type=str,
                       help='Override default experiment type')
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
    
    # Other arguments
    parser.add_argument('--save', type=int, choices=[0,1], default=0,
                       help='Whether to save the probe')
    parser.add_argument('--config', default='src/config.json',
                       help='Path to config file')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Create experiment config, updating with any provided args
    exp_config = ExperimentConfig()
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
    representations = torch.load(paths['representations']['layer'](args.layer))[args.layer]
    
    # Get labels
    labels = list(dataset[exp_config.label_field])[:len(representations)]
    
    # Split data
    train_features, train_labels, val_features, val_labels = get_train_val_split(
        representations, labels, device=device
    )
    
    # Downsample if needed
    if exp_config.downsample < 1:
        train_size = int(len(train_features) * exp_config.downsample)
        train_features = train_features[:train_size]
        train_labels = train_labels[:train_size]
    
    # Initialize model
    input_dim = representations[0].shape[-1]
    model = LinearProbe(input_dim=input_dim).to(device)
    
    # Train model
    model, metrics = train_probe(
        model, train_features, train_labels, val_features, val_labels,
        num_epochs=exp_config.num_epochs,
        learning_rate=exp_config.learning_rate,
        device=device
    )
    
    # Save results if requested
    if args.save:
        save_probe_results(
            metrics=metrics,
            model=model,
            probe_name=exp_config.get_probe_name(args.layer),
            results_name=exp_config.get_results_name(args.layer),
            probes_dir=paths['probes']['base'],
            results_dir=paths['probes']['results']
        )

if __name__ == "__main__":
    main()