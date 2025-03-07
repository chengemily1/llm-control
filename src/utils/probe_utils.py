"""Utilities for training and evaluating probes."""

import os
import random
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
from tqdm import tqdm

class LinearProbe(nn.Module):
    """Simple linear probe model."""
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

def get_train_val_split(representations, labels, split_percent=0.8, device='cuda'):
    """Split representations into training and validation sets while ensuring balanced distribution of scores."""
    # Create data dictionary with representations and their corresponding labels
    data = dict(zip(representations, labels))
    
    # Group data by score value
    score_groups = {}
    for rep, score in data.items():
        if score not in score_groups:
            score_groups[score] = []
        score_groups[score].append(rep)
    
    # Initialize train and validation sets
    train_features, train_labels = [], []
    val_features, val_labels = [], []
    
    # For each score group, split into train and validation
    for score, reps in score_groups.items():
        n_samples = len(reps)
        n_train = int(n_samples * split_percent)
        
        # Randomly sample from this score group
        train_indices = set(random.sample(range(len(reps)), n_train))
        val_indices = set(range(len(reps))) - train_indices
        
        # Split into train and validation sets
        train_reps = [reps[i] for i in train_indices]
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

def train_probe(model, train_features, train_labels, val_features, val_labels, 
                num_epochs=1000, learning_rate=0.0001, device='cuda'):
    """Train a probe model and track metrics."""
    # Setup training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    metrics = {'val_mse': [], 'val_mae': [], 'val_r2': [], 'val_epoch': []}
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        outputs = model(train_features)
        loss = criterion(outputs.squeeze(), train_labels.float())
        loss.backward()
        optimizer.step()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            outputs = model(val_features)
            val_loss = criterion(outputs.squeeze(), val_labels.float())
            predictions = outputs.squeeze()
        
        # Calculate metrics
        mse = mean_squared_error(val_labels.cpu(), predictions.cpu())
        mae = mean_absolute_error(val_labels.cpu(), predictions.cpu())
        r2 = r2_score(val_labels.cpu(), predictions.cpu())
        
        # Store metrics
        metrics['val_mse'].append(float(mse))
        metrics['val_mae'].append(float(mae))
        metrics['val_r2'].append(float(r2))
        metrics['val_epoch'].append(epoch)
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, '
                  f'MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}')
    
    return model, metrics

def save_probe_results(metrics, model, config, output_dir):
    """Save probe training results and model."""
    # Create output directories
    results_dir = os.path.join(output_dir, 'probing_results')
    probes_dir = os.path.join(output_dir, 'saved_probes')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(probes_dir, exist_ok=True)
    
    # Generate file name components
    model_name = config['model_name'].split('/')[-1]
    layer_str = f"layer_{config['layer']}"
    rs_str = f"rs{config['random_seed']}"
    ds_str = f"_downsample_{config['downsample']}" if config['downsample'] < 1 else ""
    
    # Save metrics
    metrics_file = os.path.join(
        results_dir, 
        f"{model_name}_{layer_str}_{rs_str}{ds_str}_validation_results_over_training.json"
    )
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f)
    
    # Save model if requested
    if config['save']:
        model_file = os.path.join(
            probes_dir,
            f"{model_name}_linear_probe_{layer_str}_{rs_str}{ds_str}.pt"
        )
        torch.save(model, model_file) 