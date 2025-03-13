"""Utilities for training and evaluating classification probes."""

import os
import random
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
from tqdm import tqdm
import numpy as np

def load_and_combine_representations(paths, layer):
    """Load and combine both 5000 and final representations."""
    # Load 5000 reps
    reps_5000_path = paths['representations']['layer'](layer).replace('_final.pt', '_5000.pt')
    reps_5000 = torch.load(reps_5000_path)
    if isinstance(reps_5000, dict):
        reps_5000 = reps_5000[layer]
    elif isinstance(reps_5000, list):
        reps_5000 = reps_5000[layer]
    
    # Load final reps
    reps_final_path = paths['representations']['layer'](layer)
    reps_final = torch.load(reps_final_path)
    if isinstance(reps_final, dict):
        reps_final = reps_final[layer]
    elif isinstance(reps_final, list):
        reps_final = reps_final[layer]
    
    # Combine representations
    combined_reps = torch.cat([reps_5000, reps_final], dim=0)
    print(f"\nLoaded representations:")
    print(f"5000 reps shape: {reps_5000.shape}")
    print(f"Final reps shape: {reps_final.shape}")
    print(f"Combined shape: {combined_reps.shape}\n")
    
    return combined_reps

class ClassificationProbe(nn.Module):
    """Simple linear probe model for binary classification."""
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)  # Output dim 1 for binary classification
    
    def forward(self, x):
        return self.linear(x)  # Return logits, sigmoid applied in loss/prediction

def get_train_val_split_balanced(representations, labels, split_percent=0.8, device='cuda'):
    """Split representations into training and validation sets while ensuring balanced distribution of classes."""
    # Convert boolean/float labels to integers
    labels = [int(label) if isinstance(label, (bool, np.bool_)) else label for label in labels]
    
    # Separate positive and negative examples
    positive_indices = [i for i in range(len(labels)) if labels[i]]
    negative_indices = [i for i in range(len(labels)) if not labels[i]]
    
    print(f"\nInitial class distribution:")
    print(f"Positive examples: {len(positive_indices)}")
    print(f"Negative examples: {len(negative_indices)}")
    
    # Balance the dataset by subsampling the majority class
    if len(positive_indices) > len(negative_indices):
        print("Subsampling positive examples to match negative examples")
        positive_indices = random.sample(positive_indices, len(negative_indices))
    elif len(negative_indices) > len(positive_indices):
        print("Subsampling negative examples to match positive examples")
        negative_indices = random.sample(negative_indices, len(positive_indices))
    
    print(f"\nBalanced class distribution:")
    print(f"Positive examples: {len(positive_indices)}")
    print(f"Negative examples: {len(negative_indices)}\n")
    
    # Get features for each class
    positive_features = representations[positive_indices]
    negative_features = representations[negative_indices]
    
    # Combine and get corresponding labels
    features = torch.cat([positive_features, negative_features], dim=0)
    labels = [labels[i] for i in positive_indices + negative_indices]
    
    # Create dictionary for sampling
    data = dict(zip(range(len(features)), zip(features, labels)))
    
    # Sample indices for train
    train_indices = set(random.sample(range(len(features)), int(len(features) * split_percent)))
    val_indices = set(range(len(features))) - train_indices
    
    # Split features and labels
    train_features = torch.stack([data[i][0] for i in train_indices]).float().to(device)
    train_labels = torch.tensor([data[i][1] for i in train_indices], device=device)
    
    val_features = torch.stack([data[i][0] for i in val_indices]).float().to(device)
    val_labels = torch.tensor([data[i][1] for i in val_indices], device=device)
    
    return train_features, train_labels, val_features, val_labels

def train_classification_probe(model, train_features, train_labels, val_features, val_labels, 
                             num_epochs=1000, learning_rate=0.0001, device='cuda'):
    """Train a classification probe model and track metrics."""
    # Ensure labels are float and correct shape
    train_labels = train_labels.float().unsqueeze(-1)
    val_labels = val_labels.float().unsqueeze(-1)
    
    # Print class distribution
    print("\nClass distribution:")
    print(f"Training - Class 0: {(train_labels == 0).sum().item()}, Class 1: {(train_labels == 1).sum().item()}")
    print(f"Validation - Class 0: {(val_labels == 0).sum().item()}, Class 1: {(val_labels == 1).sum().item()}\n")
    
    # Setup training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    metrics = {
        'val_accuracy': [], 
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_loss': [], 
        'val_epoch': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        outputs = model(train_features)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            outputs = model(val_features)
            val_loss = criterion(outputs, val_labels)
            
            # Get predictions using sigmoid and 0.5 threshold
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            
            # Calculate metrics
            val_labels_np = val_labels.cpu().numpy()
            predictions_np = predictions.cpu().numpy()
            
            accuracy = accuracy_score(val_labels_np, predictions_np)
            precision, recall, f1, _ = precision_recall_fscore_support(
                val_labels_np, predictions_np, average='binary'
            )
        
        # Store metrics
        metrics['val_accuracy'].append(float(accuracy))
        metrics['val_precision'].append(float(precision))
        metrics['val_recall'].append(float(recall))
        metrics['val_f1'].append(float(f1))
        metrics['val_loss'].append(float(val_loss))
        metrics['val_epoch'].append(epoch)
        
        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Loss: {val_loss:.4f}')
            print(f'  Accuracy: {accuracy:.4f}')
            print(f'  F1: {f1:.4f}')
            
            if (epoch + 1) % 1000 == 0:
                # Print probabilities for first few samples
                with torch.no_grad():
                    logits = model(val_features[:5])
                    probs = torch.sigmoid(logits)
                    print("\nProbabilities for first 5 validation samples:")
                    print("True labels:", val_labels[:5].cpu().numpy().flatten())
                    print("Predicted probs:", probs.cpu().numpy().flatten().round(3))
    
    return model, metrics

def save_classification_probe_results(metrics, model, save_name, probes_dir, results_dir, save=True):
    """Save classification probe training results and model."""
    # Create output directories
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(probes_dir, exist_ok=True)
    
    # Save metrics
    metrics_file = os.path.join(
        results_dir, 
        f"{save_name}_classification_results.json"
    )
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f)
    
    # Save model if requested
    if save:
        model_file = os.path.join(
            probes_dir,
            f"{save_name}_classification_probe.pt"
        )
        torch.save(model, model_file) 