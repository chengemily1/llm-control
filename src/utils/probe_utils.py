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
    def __init__(self, input_dim, output_dim, exp_config):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)  # This is our W matrix
        self.p = nn.Parameter(torch.randn(output_dim))  # This is our learnable p value
    
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

def train_probe(model, train_features, val_features, num_epochs=2000, learning_rate=5e-3, device='cuda'):
    """Train a probe model to learn a hyperplane Wx-p=0 that all points lie on."""
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    metrics = {
        'train_loss': [], 'val_loss': [], 'epoch': [], 
        'p_value': [], 'W_norm': [], 'W_stats': [],
        'mean_distance': [], 'max_distance': [], 'std_distance': [],
        'percent_on_hyperplane': [], 'num_points_on_hyperplane': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        
        # Get distances to hyperplane
        train_distances = model.get_distance_to_hyperplane(train_features)
        
        # Direct distance minimization loss
        loss = torch.norm(train_distances)
        
        # Add L2 regularization on W to prevent degenerate solutions
        W_norm = torch.norm(model.linear.weight)
        loss = loss + 1e-4 * (W_norm - 1.0) ** 2  # Encourage W to have norm close to 1
        
        loss.backward()
        optimizer.step()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_distances = model.get_distance_to_hyperplane(val_features)
            val_loss = torch.norm(val_distances)
            
            # Get statistics about W
            W = model.linear.weight
            W_norm = torch.norm(W).item()
            W_mean = torch.mean(W).item()
            W_std = torch.std(W).item()
            W_min = torch.min(W).item()
            W_max = torch.max(W).item()
            
            # Calculate distance statistics
            mean_dist = torch.mean(train_distances).item()
            max_dist = torch.max(torch.abs(train_distances)).item()
            std_dist = torch.std(train_distances).item()
            
            # Count points that are "exactly" on hyperplane
            exact_threshold = 1e-6
            # points_on_hyperplane = torch.sum(torch.abs(train_distances) < exact_threshold).item()
            # total_points = train_distances.shape[0]
            # percent_on_hyperplane = 100 * points_on_hyperplane / total_points
        
        # Store all metrics
        metrics['train_loss'].append(float(loss))
        metrics['val_loss'].append(float(val_loss))
        metrics['epoch'].append(epoch)
        metrics['p_value'].append(model.p.detach().cpu().numpy().tolist())
        metrics['W_norm'].append(W_norm)
        metrics['W_stats'].append({
            'mean': W_mean,
            'std': W_std,
            'min': W_min,
            'max': W_max
        })
        metrics['mean_distance'].append(mean_dist)
        metrics['max_distance'].append(max_dist)
        metrics['std_distance'].append(std_dist)
        # metrics['percent_on_hyperplane'].append(percent_on_hyperplane)
        # metrics['num_points_on_hyperplane'].append(points_on_hyperplane)
        
        if (epoch + 1) % 10 == 0:
            input()
            print(f'Epoch {epoch+1}/{num_epochs}')
            print(f'Loss: {loss:.8f}, Val Loss: {val_loss:.8f}')
            print(f'Mean Distance: {mean_dist:.8f}, Max Abs Distance: {max_dist:.8f}')
            # print(f'Std Distance: {std_dist:.8f}, p: {model.p.detach().cpu().numpy().tolist()}')
            print(f'W norm: {W_norm:.8f}')
            print(f'W stats - mean: {W_mean:.8f}, std: {W_std:.8f}, min: {W_min:.8f}, max: {W_max:.8f}')
            # print(f'Points exactly on hyperplane: {points_on_hyperplane}/{total_points} ({percent_on_hyperplane:.2f}%)')
    
    return model, metrics, train_features, val_features

def save_probe_results(metrics, model, config, output_dir, train_features=None, val_features=None):
    """Save probe training results, model, and representations.
    
    Args:
        metrics: Dictionary containing training metrics
        model: Trained probe model
        config: Configuration dictionary containing model_name, layer, random_seed, etc.
        output_dir: Base output directory
        train_features: Training set representations (optional)
        val_features: Validation set representations (optional)
    """
    # Create experiment-specific paths
    experiment_dir = os.path.join(output_dir, 'experiments', 'gms8k')
    results_dir = os.path.join(experiment_dir, 'probing_results')
    probes_dir = os.path.join(experiment_dir, 'saved_probes')
    reps_dir = os.path.join(experiment_dir, 'saved_representations')
    
    # Create directories
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(probes_dir, exist_ok=True)
    os.makedirs(reps_dir, exist_ok=True)
    
    # Generate file name components
    model_name = config['model_name'].split('/')[-1]
    layer_str = f"layer_{config['layer']}"
    rs_str = f"rs{config['random_seed']}"
    ds_str = f"_downsample_{config['downsample']}" if config.get('downsample', 1.0) < 1 else ""
    base_name = f"{model_name}_{layer_str}_{rs_str}{ds_str}"
    
    # Save final model state and hyperplane parameters
    final_state = {
        'W': model.linear.weight.detach().cpu().numpy().tolist(),
        'p': model.p.detach().cpu().numpy().tolist(),
        'W_norm': torch.norm(model.linear.weight).item(),
        'final_metrics': {
            'train_loss': metrics['train_loss'][-1],
            'val_loss': metrics['val_loss'][-1],
            'mean_distance': metrics['mean_distance'][-1],
            'max_distance': metrics['max_distance'][-1],
            'std_distance': metrics['std_distance'][-1],
            'percent_on_hyperplane': metrics['percent_on_hyperplane'][-1],
            'num_points_on_hyperplane': metrics['num_points_on_hyperplane'][-1]
        }
    }
    
    # Save detailed training history
    metrics['hyperplane_params'] = final_state
    metrics_file = os.path.join(results_dir, f"{base_name}_training_history.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved training history to {metrics_file}")
    
    # Save model
    model_file = os.path.join(probes_dir, f"{base_name}_probe.pt")
    torch.save(model.state_dict(), model_file)  # Save state dict instead of full model
    print(f"Saved probe model to {model_file}")
    
    # Save representations if provided
    if train_features is not None:
        train_reps_file = os.path.join(reps_dir, f"{base_name}_train_representations.pt")
        torch.save(train_features.cpu(), train_reps_file)
        print(f"Saved training representations to {train_reps_file}")
    
    if val_features is not None:
        val_reps_file = os.path.join(reps_dir, f"{base_name}_val_representations.pt")
        torch.save(val_features.cpu(), val_reps_file)
        print(f"Saved validation representations to {val_reps_file}") 