###################  Setup & packages ################### 
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pdb
import os
from config import YOUR_PATH, YOUR_TOKEN

###################  Script arguments ###################
parser = argparse.ArgumentParser(description='Train probe')
# Model selection
parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3-8B")
# Experiment selection
parser.add_argument('--experiment', type=str, default='elix')
# Training settings
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--layer', type=int)
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--downsample', type=float, default=1)
# Directory management
parser.add_argument('--save', type=int, choices=[0,1], default=1, help='whether to save the probe')
parser.add_argument('--path_to_data', type=str, default=YOUR_PATH)
args = parser.parse_args()

if args.experiment == 'elix':
    label = 'score'

###################  Learning configuration ###################
ACCESS_TOKEN=YOUR_TOKEN
random.seed(args.random_seed)
device = 'cuda:0' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'

###################  Load data ###################
file_path = f"{args.path_to_data}/elix_generations_gpt4omini_pref/train_shuffled_balanced.csv"
dataset = pd.read_csv(file_path)

###################  Load representations ###################
# Load representations
data_path = args.path_to_data
# List all relevant files
#files = [f for f in os.listdir(data_path) if f.startswith(args.model_name.split("/")[-1]) and f.endswith('new.pt')]
# Sort files numerically based on the part number
#files.sort(key=lambda x: int(x.split('_part_')[-1].split('.')[0]) if '_part_' in x else float('inf'))
# Initialize an empty tensor for representations
#representations = None
# Load the first three files and append their contents
#for f in files[0:1]:  # Only take the first file
fpath = os.path.join(data_path, 'experiments/elix/saved_reps/Meta-Llama-3-8B_reps_final.pt')
representations = torch.load(fpath)[args.layer]

###################  Data preparation ###################
def get_train_val(layer, split_percent=0.8):
    """
    Split representations into training and validation set while ensuring balanced distribution of scores.
    """
    labels = list(dataset[label]) # Get the labels
    
    # Only take labels corresponding to the first len(layer) components
    labels = labels[:len(layer)]  # Update to slice labels based on the length of layer
    
    # Create data dictionary with representations and their corresponding labels
    data = dict(zip(layer, labels))
    
    # Group data by score value
    score_groups = {}
    for rep, score in data.items():
        if score not in score_groups:
            score_groups[score] = []
        score_groups[score].append(rep)
    
    # Initialize train and validation sets
    train_features = []
    train_labels = []
    val_features = []
    val_labels = []
    
    # For each score group, split into train and validation
    for score, reps in score_groups.items():
        n_samples = len(reps)
        n_train = int(n_samples * split_percent)
        
        # Randomly sample from this score group
        train_reps = random.sample(reps, n_train)
        # Create a set of train indices for faster lookup
        train_indices = set(range(len(reps)))
        train_indices = set(random.sample(list(train_indices), n_train))
        val_indices = set(range(len(reps))) - train_indices
        
        # Get validation reps using indices
        val_reps = [reps[i] for i in val_indices]
        
        train_features.extend(train_reps)
        train_labels.extend([score] * len(train_reps))
        val_features.extend(val_reps)
        val_labels.extend([score] * len(val_reps))
    
    # Convert to tensors and move to device
    train_features = torch.stack(train_features).float().to(device)
    val_features = torch.stack(val_features).float().to(device)
    
    train_labels = torch.tensor(train_labels, device=device)
    val_labels = torch.tensor(val_labels, device=device)
    
    return train_features, train_labels, val_features, val_labels

train_features, train_labels, val_features, val_labels = get_train_val(representations)

# Loop for downsizing traning data (if needed)
if args.downsample < 1:
    train_features = train_features[:int(len(train_features) * args.downsample)]
    train_labels = train_labels[:int(len(train_labels) * args.downsample)]

###################  Training parameters ###################
pretrained_model_output_dim = representations[10].shape[-1] # take a random layer e.g. layer 10 and get the output dim
num_epochs = 1000
learning_rate = 0.0001
# Define linear regressor
linear_probe = nn.Linear(pretrained_model_output_dim, 1)  # Output dim of pre-trained model -> single output
linear_probe.to(device)
# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(linear_probe.parameters(), lr=learning_rate)

mse_scores, mae_scores, r2_scores = [], [], []

###################  Training loop ###################
for epoch in range(num_epochs):
    # Training phase
    linear_probe.train() # Set the model to training mode
    optimizer.zero_grad() # Reset gradients to zero
    outputs = linear_probe(train_features) # Forward pass
    loss = criterion(outputs.squeeze(), train_labels.float()) # Compute the loss
    # Backward pass
    loss.backward() # Compute the gradients
    optimizer.step() # Update model parameters
    # Validation 
    linear_probe.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculation
        outputs = linear_probe(val_features) # Forward pass
        val_loss = criterion(outputs.squeeze(), val_labels.float()) # Compute the loss
        predictions = outputs.squeeze()
    
    # Metric calculation
    mse = mean_squared_error(val_labels.cpu(), predictions.cpu())
    mae = mean_absolute_error(val_labels.cpu(), predictions.cpu())
    r2 = r2_score(val_labels.cpu(), predictions.cpu())
    
    mse_scores.append(mse)
    mae_scores.append(mae)
    r2_scores.append(r2)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}')

###################  Save the results ###################
# Save validation metrics
results = {
    'val_mse': [float(x) for x in mse_scores], 
    'val_mae': [float(x) for x in mae_scores], 
    'val_r2': [float(x) for x in r2_scores], 
    'val_epoch': list(range(num_epochs))
}
# Create the probing_results directory if it doesn't exist
results_dir = os.path.join(args.path_to_data, 'experiments/elix/probing_results')
probes_dir = os.path.join(args.path_to_data, 'experiments/elix/saved_probes')
os.makedirs(results_dir, exist_ok=True)  # This will create the directory if it doesn't exist
# Save results to file
os.makedirs(probes_dir, exist_ok=True)
with open(args.path_to_data + f'/experiments/elix/probing_results/{args.model_name.split("/")[-1]}_layer_{args.layer}_rs{args.random_seed}{f"_downsample_{args.downsample}" if args.downsample < 1 else ""}_validation_results_over_training.json', 'w') as f:
    json.dump(results, f)
# Save probe
if args.save:
    torch.save(linear_probe, f'{args.path_to_data}/experiments/elix/saved_probes/{args.model_name.split("/")[-1]}_linear_probe_layer_{args.layer}_rs{args.random_seed}{f"_downsample_{args.downsample}" if args.downsample < 1 else ""}.pt')