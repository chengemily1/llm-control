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
from sklearn.metrics import f1_score
import pdb
import os

###################  Script arguments ###################
parser = argparse.ArgumentParser(description='Train probe')
# Model selection
parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3-8B")
# Experiment selection
parser.add_argument('--experiment', type=str, default='reasoning')
# Training settings
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--layer', type=int)
parser.add_argument('--random_seed', type=int)
parser.add_argument('--downsample', type=float, default=1)
# Directory management
parser.add_argument('--save', type=int, choices=[0,1], default=1, help='whether to save the probe')
parser.add_argument('--path_to_data', type=str, default='../../shared_data/llm_control/')
args = parser.parse_args()

if args.experiment == 'reasoning':
    label = 'correct'

###################  Learning configuration ###################
ACCESS_TOKEN='YOUR TOKEN'
random.seed(args.random_seed)
device = 'cuda:0' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'

###################  Load data ###################
file_path = f"{args.path_to_data}/train_shuffled_balanced.csv"
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
fpath = os.path.join(data_path, 'Meta-Llama-3-8B_reps_part_5000_new.pt')
loaded_reps = torch.load(fpath)[args.layer]
#if representations is None:
representations = loaded_reps  # Initialize representations with the first loaded tensor
#else:
#    representations = torch.cat((representations, loaded_reps), dim=0)  # Append the loaded tensor

###################  Data preparation ###################
def get_train_val(layer, split_percent=0.8):
    """
    Split representations into training and validation set.
    """
    labels = list(dataset[label]) # Get the labels
    # If correct is 0, then it is a correct answer
    labels = [1 - int(label) if type(label) == bool else 1 - label for label in labels]  

    # Only take labels corresponding to the first len(layer) components
    labels = labels[:len(layer)]  # Update to slice labels based on the length of layer

    toxic = [i for i in range(len(labels)) if labels[i]]
    # Calculate the target number of toxic samples
    target_toxic_count = len(layer) // 2 - max(0, len(toxic) - (len(layer) // 2))
    # Drop random elements until len(toxic) is target_toxic_count to ensure balanced dataset
    if len(toxic) > target_toxic_count:
        toxic = random.sample(toxic, target_toxic_count)  # Randomly sample to reduce size
    untoxic = random.sample([i for i in range(len(labels)) if not labels[i]], len(toxic))
    
    layer_toxic = layer[toxic,:]
    layer_untoxic = layer[untoxic,:]

    layer = torch.concat([layer_toxic, layer_untoxic], dim=0)
    labels = [labels[n] for array in [toxic, untoxic] for n in array]

    data = dict(zip(layer, labels))

    train_features = random.sample(list(data.keys()), int(len(data) * split_percent))
    train_labels = [data[train_feat] for train_feat in train_features]
    
    val_features = [elt for elt in data if elt not in set(train_features)]
    val_labels = [data[val_feat] for val_feat in val_features]

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
num_classes = 2
num_epochs = 1000
learning_rate = 0.0001
# Define linear probe
linear_probe = nn.Linear(pretrained_model_output_dim, num_classes)  # Output dim of pre-trained model -> num classes
linear_probe.to(device)
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(linear_probe.parameters(), lr=learning_rate)

accs, f1s = [], []

###################  Training loop ###################
for epoch in range(num_epochs):
    # Training phase
    linear_probe.train() # Set the model to training mode
    optimizer.zero_grad() # Reset gradients to zero
    outputs = linear_probe(train_features) # Forward pass
    loss = criterion(outputs, train_labels) # Compute the loss
    # Backward pass
    loss.backward() # Compute the gradients
    optimizer.step() # Update model parameters
    # Validation 
    linear_probe.eval() # Set model to evaluation mode
    val_loss = 0 # Initialize
    correct = 0 # Initialize
    total = 0 # Initialize
    with torch.no_grad(): # Disable gradient calculation
        outputs = linear_probe(val_features) # Forward pass
        val_loss += criterion(outputs, val_labels) # Compute the loss
        _, predicted = outputs.max(1) # Get predicted labels
        total += val_labels.size(0) # Updates total number of validation samples
        # if not args.continuous_tune: 
        correct += predicted.eq(val_labels).sum().item() # Compare predicted vs true labels
        binary_val = val_labels
    # Metric calculation
    val_loss /= len(val_features) # Averages the validation loss over the number of validation samples
    accuracy = 100. * correct / total # Calculates the accuracy as a percentage
    f1 = f1_score(binary_val.cpu(), predicted.cpu()) # Computes the F1 score, which is a measure of a model's accuracy that considers both precision and recall
    f1s.append(f1)
    accs.append(accuracy)
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%, F1: {f1:.2f}')

###################  Save the results ###################
# Save validation accuracy, f1
results = {'val_acc': accs, 'val_f1': f1s, 'val_epoch': list(range(num_epochs))}
# Create the probing_results directory if it doesn't exist
#path_to_data = '/experiments/reasoning'
results_dir = os.path.join(args.path_to_data, 'probing_results')
probes_dir = os.path.join(args.path_to_data, 'saved_probes')
os.makedirs(results_dir, exist_ok=True)  # This will create the directory if it doesn't exist
# Save results to file
os.makedirs(probes_dir, exist_ok=True)
with open(args.path_to_data + f'/probing_results/{args.model_name.split("/")[-1]}_layer_{args.layer}_rs{args.random_seed}{f"_downsample_{args.downsample}" if args.downsample < 1 else ""}_validation_results_over_training.json', 'w') as f:
    json.dump(results, f)
# Save probe
if args.save:
    torch.save(linear_probe, f'{args.path_to_data}/saved_probes/{args.model_name.split("/")[-1]}_linear_probe_layer_{args.layer}_rs{args.random_seed}{f"_downsample_{args.downsample}" if args.downsample < 1 else ""}.pt')