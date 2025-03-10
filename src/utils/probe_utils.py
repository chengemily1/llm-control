"""Utilities for training and evaluating probes."""

import os
import random
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
from tqdm import tqdm

class LinearProbe(nn.Module):
    """Linear probe that learns a hyperplane Wx-p=0."""
    def __init__(self, input_dim, exp_config):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False)  # This is our W matrix
        self.p = nn.Parameter(torch.randn(1))  # This is our learnable p value
    
    def forward(self, x):
        return torch.matmul(x, self.linear.weight.t()).squeeze(-1)  # Returns Wx
    
    def get_distance_to_hyperplane(self, x):
        """Get the signed distance from points to the hyperplane Wx-p=0."""
        Wx = self.forward(x)
        return Wx - self.p  # Distance is Wx-p

def get_train_val_split(representations, split_percent=0.8, device='cuda'):
    """Split representations into training and validation sets."""
    n_samples = len(representations)
    n_train = int(n_samples * split_percent)
    
    # Randomly shuffle indices
    indices = list(range(n_samples))
    random.shuffle(indices)
    
    # Split into train and validation sets
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    # Split features
    train_features = torch.stack([representations[i] for i in train_indices]).float().to(device)
    val_features = torch.stack([representations[i] for i in val_indices]).float().to(device)
    
    return train_features, val_features

def train_probe(model, train_features, val_features, num_epochs=1000, learning_rate=0.0001, device='cuda'):
    """Train a probe model to learn a hyperplane Wx-p=0 that all points lie on."""
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    metrics = {'train_loss': [], 'val_loss': [], 'epoch': [], 'p_value': [], 'W_norm': [], 'W_stats': []}
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        
        # Get distances to hyperplane
        train_distances = model.get_distance_to_hyperplane(train_features)
        
        # Use higher power to more strictly enforce points lying on hyperplane
        # Power of 4 will more heavily penalize points far from hyperplane
        loss = torch.mean(train_distances ** 4)
        
        # Add L2 regularization on W to prevent degenerate solutions
        W_norm = torch.norm(model.linear.weight)
        loss = loss + 1e-4 * (W_norm - 1.0) ** 2  # Encourage W to have norm close to 1
        
        loss.backward()
        optimizer.step()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_distances = model.get_distance_to_hyperplane(val_features)
            val_loss = torch.mean(val_distances ** 4)
            
            # Get statistics about W
            W = model.linear.weight
            W_norm = torch.norm(W).item()
            W_mean = torch.mean(W).item()
            W_std = torch.std(W).item()
            W_min = torch.min(W).item()
            W_max = torch.max(W).item()
            
            # Count points that are "exactly" on hyperplane (within numerical precision)
            exact_threshold = 1e-6
            points_on_hyperplane = torch.sum(torch.abs(train_distances) < exact_threshold).item()
            total_points = train_distances.shape[0]
            percent_on_hyperplane = 100 * points_on_hyperplane / total_points
        
        # Store metrics
        metrics['train_loss'].append(float(loss))
        metrics['val_loss'].append(float(val_loss))
        metrics['epoch'].append(epoch)
        metrics['p_value'].append(float(model.p))
        metrics['W_norm'].append(W_norm)
        metrics['W_stats'].append({
            'mean': W_mean,
            'std': W_std,
            'min': W_min,
            'max': W_max
        })
        
        if (epoch + 1) % 100 == 0:
            mean_dist = torch.mean(train_distances)
            max_dist = torch.max(torch.abs(train_distances))
            std_dist = torch.std(train_distances)
            print(f'Epoch {epoch+1}/{num_epochs}')
            print(f'Loss: {loss:.8f}, Val Loss: {val_loss:.8f}')
            print(f'Mean Distance: {mean_dist:.8f}, Max Abs Distance: {max_dist:.8f}')
            print(f'Std Distance: {std_dist:.8f}, p: {model.p.item():.8f}')
            print(f'W norm: {W_norm:.8f}')
            print(f'W stats - mean: {W_mean:.8f}, std: {W_std:.8f}, min: {W_min:.8f}, max: {W_max:.8f}')
            print(f'Points exactly on hyperplane: {points_on_hyperplane}/{total_points} ({percent_on_hyperplane:.2f}%)')
    
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
    
    # Save metrics and final target vector
    metrics['final_target_vector'] = model.linear.weight.detach().cpu().numpy().tolist()
    
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