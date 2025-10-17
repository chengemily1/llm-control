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
from torcheval.metrics.functional import r2_score

from control_wrapper import *

parser = argparse.ArgumentParser(description='Train probe')

# Data selection
parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3-8B")
parser.add_argument('--experiment', type=str, choices=['toxicity', 'sentiment', 'formality'])
parser.add_argument('--objective', type=str, default='classification', choices=['regression', 'classification'])
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--num_epochs', type=int, default=20000)
parser.add_argument('--learning_rate', type=float, default=0.005)
parser.add_argument('--layer', type=int)
parser.add_argument('--continuous_label', type=int, default=0)
parser.add_argument('--random_seed', type=int)
parser.add_argument('--downsample', type=float, default=1.0)
parser.add_argument('--map_to_target_space', type=str, choices=['tanh', 'sigmoid', 'identity'])
parser.add_argument('--save', type=int, choices=[0,1], default=0, help='whether to save the probe')
parser.add_argument('--config', type=str, help='/path/to/config.json')

args = parser.parse_args()
print(args)

### CONFIG and LOADING
with open(args.config, 'r') as f:
    CONFIG = json.load(f)

ACCESS_TOKEN = CONFIG['hf_access_token']
YOUR_PATH = CONFIG['path']

random.seed(args.random_seed)

DATASET_PATH = {
    'toxicity': f'{YOUR_PATH}/jigsaw-toxic-comment-classification-challenge/train_shuffled_balanced.csv',
    'sentiment': f'{YOUR_PATH}/sentiment-constraint-set/train_shuffled_balanced.csv',
    'formality': f'{YOUR_PATH}/formality-constraint-set/train_shuffled_balanced.csv'
}
args.path_to_reps = f'{YOUR_PATH}/experiments/{args.experiment}/'

device = 'cuda:0' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
dataset = pd.read_csv(DATASET_PATH[args.experiment]) # we shuffled the ds in saving the reps

if args.experiment == 'sentiment':
    label = 'negative' if not args.continuous_label else 'p_negative'
elif args.experiment == 'toxicity':
    label = 'toxic' if not args.continuous_label else 'p_toxic'
elif args.experiment == 'formality':
    label = 'auto_formality_score'
    cont_label = 'formal'

def get_train_val(layer, split_percent=0.8):
    """
    Split representations into training and validation set.
    """
    labels = list(dataset[label]) # Get the binary labels
    labels = [int(label) if type(label)==bool else label for label in labels]
    layer = layer[:30000,:]
    labels = labels[:30000]

    if args.objective == 'classification':
        if type(labels[0]) == bool:
            labels = [int(label) for label in labels]
        toxic = [i for i in range(len(labels)) if labels[i] > 0.5]
        untoxic = [i for i in range(len(labels)) if labels[i] <= 0.5]
        min_sample = min(len(toxic), len(untoxic))
        untoxic = random.sample(untoxic, min_sample)
        toxic = random.sample(toxic, min_sample)

    elif args.objective == 'regression':
        if args.experiment == 'formality':
            toxic = [i for i in range(len(labels)) if labels[i] < 0]
            untoxic = [i for i in range(len(labels)) if labels[i] >= 0]
        else:
            toxic = [i for i in range(len(labels)) if labels[i] > 0.5]
            untoxic = random.sample([i for i in range(len(labels)) if labels[i] <= 0.5], len(toxic))

    if args.objective == 'regression':
        labels = dataset[cont_label] # continuous scores

    layer_toxic = layer[toxic,:]
    layer_untoxic = layer[untoxic,:]

    layer = torch.concat([layer_toxic, layer_untoxic], dim=0)
    labels = [labels[n] for array in [toxic, untoxic] for n in array]

    data = dict(zip(layer, labels))

    train_features = random.sample(list(data.keys()), int(len(data) * split_percent))

    if args.objective == 'classification':
        train_labels = [data[train_feat] for train_feat in train_features]
    elif args.objective == 'regression':
        # TODO
        train_labels = [data[train_feat] for train_feat in train_features]
        # train_labels = [[data[train_feat], 1 - data[train_feat]] for train_feat in train_features]

    val_features = [elt for elt in data if elt not in set(train_features)]

    if args.objective == 'classification':
        val_labels = [data[val_feat] for val_feat in val_features]
    elif args.objective == 'regression':
        val_labels = [data[val_feat] for val_feat in val_features]

        # val_labels = [[data[val_feat], 1 - data[val_feat]] for val_feat in val_features] # p(formal) or p(toxic), 1-p

    train_features = torch.stack(train_features).float().to(device)
    val_features = torch.stack(val_features).float().to(device)

    train_labels = torch.tensor(train_labels, device=device)
    val_labels = torch.tensor(val_labels, device=device)

    print('Training set size: ', len(train_features))
    print('Validation set size: ', len(val_features))

    return train_features, train_labels, val_features, val_labels

# Load representations
fpath = args.path_to_reps + f'saved_reps/{args.model_name.split("/")[-1]}_reps.pt'
representations = torch.load(fpath)[args.layer]
representations = representations.to(device)

#### TRAIN
num_epochs = args.num_epochs
learning_rate = 0.00005

# Iterate over layers
train_features, train_labels, val_features, val_labels = get_train_val(representations)

if args.downsample < 1:
    train_features = train_features[:int(len(train_features) * args.downsample)]
    train_labels = train_labels[:int(len(train_labels) * args.downsample)]

# Define linear probe
pretrained_model_output_dim = representations[10].shape[-1] # take a random layer e.g. layer 10 and get the output dim
linear_probe = nn.Linear(pretrained_model_output_dim, 1)  # Output dim of pre-trained model -> R
linear_probe.to(device)
print(linear_probe)

# Define loss function and optimizer
if args.objective == 'classification':
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
elif args.objective == 'regression':
    criterion = nn.MSELoss(reduction='mean')

optimizer = torch.optim.Adam(linear_probe.parameters(), lr=learning_rate)

accs, f1s, r2s, losses = [], [], [], []

# if args.objective == 'classification':
train_labels = train_labels.float().unsqueeze(-1)
val_labels = val_labels.float().unsqueeze(-1)

# Training loop
early_stop = False
best_loss = float('inf')

for epoch in range(num_epochs):
    linear_probe.train()
    optimizer.zero_grad()
    outputs = linear_probe(train_features)
    if args.objective == 'regression':
        nonlinearity = MAP[args.map_to_target_space]
        outputs = nonlinearity(outputs)
        
    loss = criterion(outputs, train_labels)

    # Backward pass
    loss.backward()
    optimizer.step()

    # Validation
    linear_probe.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        outputs = linear_probe(val_features)
        if args.objective == 'regression':
            nonlinearity = MAP[args.map_to_target_space]
            outputs = nonlinearity(outputs)
        val_loss = criterion(outputs, val_labels).float().item()

        if len(losses) > 2000:
            if val_loss > best_loss:
                early_stop = True
            else:
                best_loss = val_loss

        losses.append(val_loss)

        total = val_labels.size(0)

        if args.objective == 'classification':
            predicted = (nn.functional.sigmoid(outputs) > 0.5).long().float() # convert to 0s and 1s
            binary_val = (val_labels > 0.5).long().float()
            correct = predicted.eq(binary_val).sum().item()

            accuracy = 100. * correct / total

            f1 = f1_score(binary_val.cpu(), predicted.cpu())
            f1s.append(f1)
            accs.append(accuracy)

            print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%, F1: {f1:.2f}')

        elif args.objective == 'regression':
            # predicted = None
            # binary_val = (val_labels[:,0] < 0.5).long()
            # match = (binary_val.eq(predicted)).sum().item()
            # correct += match
            r2 = r2_score(outputs, val_labels).float().item()
            r2s.append(r2)

            print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, R2: {r2:.2f}')

    if early_stop:
        break

# Save validation accuracy, metrics
results = {'val_loss': losses, 'val_acc': accs, 'val_f1': f1s, 'val_r2': r2s}

with open(args.path_to_reps + f'probing_results/{args.model_name.split("/")[-1]}_layer_{args.layer}_rs{args.random_seed}{f"_downsample_{args.downsample}" if args.downsample < 1 else ""}_validation_results_over_training.json', 'w') as f:
    json.dump(results, f)

# Save probe
if args.save:
    torch.save(linear_probe, f'{args.path_to_reps}saved_probes/{args.model_name.split("/")[-1]}_linear_probe_layer_{args.layer}_rs{args.random_seed}{f"_downsample_{args.downsample}" if args.downsample < 1 else ""}.pt')
