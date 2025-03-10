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
                num_epochs=1000, learning_rate=0.0001, device='cuda', 
                patience=50, early_stopping_metric='val_mse', min_delta=0.0001):
    """Train a probe model and track metrics.
    
    Args:
        model: The probe model to train
        train_features: Training features
        train_labels: Training labels
        val_features: Validation features
        val_labels: Validation labels
        num_epochs: Maximum number of epochs to train for
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cuda' or 'cpu')
        patience: Number of epochs to wait for improvement before stopping
        early_stopping_metric: Metric to monitor for early stopping ('val_mse', 'val_mae', or 'val_r2')
        min_delta: Minimum change in monitored metric to qualify as improvement
        
    Returns:
        Tuple of (best_model, metrics)
    """
    # Setup training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    metrics = {'val_mse': [], 'val_mae': [], 'val_r2': [], 'val_epoch': []}
    
    # Early stopping setup
    best_metric_value = float('inf') if early_stopping_metric != 'val_r2' else float('-inf')
    best_epoch = 0
    best_model_state = None
    counter = 0
    
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
        
        # Get current metric value for early stopping
        if early_stopping_metric == 'val_mse':
            current_metric = mse
        elif early_stopping_metric == 'val_mae':
            current_metric = mae
        elif early_stopping_metric == 'val_r2':
            current_metric = r2
        else:
            raise ValueError(f"Unknown early stopping metric: {early_stopping_metric}")
        
        # Check if this is the best model so far
        improved = False
        if early_stopping_metric == 'val_r2':  # For R², higher is better
            if current_metric > best_metric_value + min_delta:
                improved = True
        else:  # For MSE and MAE, lower is better
            if current_metric < best_metric_value - min_delta:
                improved = True
                
        if improved:
            best_metric_value = current_metric
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
        
        # Print progress
        if (epoch + 1) % 100 == 0 or counter == patience:
            print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, '
                  f'MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}')
            if counter > 0:
                print(f'Early stopping counter: {counter}/{patience}')
        
        # Check early stopping condition
        if counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs. Best epoch was {best_epoch+1}.')
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'Restored model from best epoch ({best_epoch+1})')
    
    # Add best epoch to metrics
    metrics['best_epoch'] = best_epoch
    
    return model, metrics

def save_probe_results(metrics, model, save_name, probes_dir, results_dir, save=True):
    """Save probe training results and model."""
    # Create output directories
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(probes_dir, exist_ok=True)
    
    # Save metrics
    metrics_file = os.path.join(
        results_dir, 
        f"{save_name}_validation_results_over_training.json"
    )
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f)
    
    # Save model if requested
    if save:
        model_file = os.path.join(
            probes_dir,
            f"{save_name}_linear_probe.pt"
        )
        torch.save(model, model_file) 