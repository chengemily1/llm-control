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


parser = argparse.ArgumentParser(description='Train probe')

# Data selection
parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument('--experiment', type=str, choices=['toxicity', 'sentiment', 'formality'])
# parser.add_argument('--dataset_name', type=str, default='/home/echeng/llm-control/jigsaw-toxic-comment-classification-challenge')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda')
# parser.add_argument('--path_to_reps', type=str, default='/home/echeng/llm-control/experiments/toxicity/')
parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--layer', type=int)
parser.add_argument('--continuous_tune', type=int, default=1)
parser.add_argument('--random_seed', type=int)
parser.add_argument('--downsample', type=float, default=1)
parser.add_argument('--save', type=int, choices=[0,1], default=0, help='whether to save the probe')

args = parser.parse_args()
print(args)

ACCESS_TOKEN='YOUR TOKEN'
random.seed(args.random_seed)

DATASET_PATH = {
    'toxicity': '/home/echeng/llm-control/jigsaw-toxic-comment-classification-challenge/train_shuffled_balanced.csv',
    'sentiment': '/home/echeng/llm-control/sentiment-constraint-set/train_shuffled_balanced.csv',
    'formality': '/home/echeng/llm-control/formality-constraint-set/train_shuffled_balanced.csv',
}
args.path_to_reps = f'/home/echeng/llm-control/experiments/{args.experiment}/'

# model.eval()
device = 'cuda:0' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
dataset = pd.read_csv(DATASET_PATH[args.experiment]) # we shuffled the ds in saving the reps

if args.experiment == 'sentiment':
    label = 'negative'
elif args.experiment == 'toxicity':
    label = 'toxic'
elif args.experiment == 'formality':
    label = 'formal'

def get_train_val(layer, split_percent=0.8):
    """
    Split representations into training and validation set.
    """
    labels = list(dataset[label]) # Get the labels
    labels = [int(label) if type(label)==bool else label for label in labels]
    if not args.continuous_tune:
        toxic = [i for i in range(len(labels)) if labels[i]]
        untoxic = random.sample([i for i in range(len(labels)) if not labels[i]], len(toxic))
    if args.continuous_tune:
        toxic = [i for i in range(len(labels)) if labels[i] > 0.5]
        untoxic = random.sample([i for i in range(len(labels)) if labels[i] <= 0.5], len(toxic))

    layer_toxic = layer[toxic,:]
    layer_untoxic = layer[untoxic,:]

    layer = torch.concat([layer_toxic, layer_untoxic], dim=0)
    labels = [labels[n] for array in [toxic, untoxic] for n in array]

    data = dict(zip(layer, labels))

    train_features = random.sample(list(data.keys()), int(len(data) * split_percent))
    if not args.continuous_tune:
        train_labels = [data[train_feat] for train_feat in train_features]
    else:
        train_labels = [[data[train_feat], 1 - data[train_feat]] for train_feat in train_features]

    val_features = [elt for elt in data if elt not in set(train_features)]
    if not args.continuous_tune:
        val_labels = [data[val_feat] for val_feat in val_features]
    else:
        val_labels = [[data[val_feat], 1 - data[val_feat]] for val_feat in val_features] # p(formal) or p(toxic), 1-p

    train_features = torch.stack(train_features).float().to(device)
    val_features = torch.stack(val_features).float().to(device)

    train_labels = torch.tensor(train_labels, device=device)
    val_labels = torch.tensor(val_labels, device=device)

    return train_features, train_labels, val_features, val_labels

# Load representations
fpath = args.path_to_reps + f'saved_reps/{args.model_name.split("/")[-1]}_reps.pt'
print(fpath)
representations = torch.load(fpath)[args.layer]

# Get linear probe
pretrained_model_output_dim = representations[10].shape[-1] # take a random layer e.g. layer 10 and get the output dim
num_classes = 2

# TRAIN
num_epochs = 1000
learning_rate = 0.0001

# Iterate over layers
train_features, train_labels, val_features, val_labels = get_train_val(representations)

if args.downsample < 1:
    train_features = train_features[:int(len(train_features) * args.downsample)]
    train_labels = train_labels[:int(len(train_labels) * args.downsample)]

# Define linear probe
linear_probe = nn.Linear(pretrained_model_output_dim, num_classes)  # Output dim of pre-trained model -> num classes
linear_probe.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(linear_probe.parameters(), lr=learning_rate)

accs, f1s = [], []

# Training loop
for epoch in range(num_epochs):
    linear_probe.train()
    optimizer.zero_grad()
    outputs = linear_probe(train_features)
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
        val_loss += criterion(outputs, val_labels)
        _, predicted = outputs.max(1)
        total += val_labels.size(0)

        if not args.continuous_tune:
            correct += predicted.eq(val_labels).sum().item()
            binary_val = val_labels
        else:
            binary_val = (val_labels[:,0] < 0.5).long()
            match = (binary_val.eq(predicted)).sum().item()
            correct += match

    val_loss /= len(val_features)

    accuracy = 100. * correct / total
    # pdb.set_trace()
    f1 = f1_score(binary_val.cpu(), predicted.cpu())
    f1s.append(f1)
    accs.append(accuracy)
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%, F1: {f1:.2f}')

# Save validation accuracy, f1
results = {'val_acc': accs, 'val_f1': f1s, 'val_epoch': list(range(num_epochs))}
with open(args.path_to_reps + f'/probing_results/{args.model_name.split("/")[-1]}_layer_{args.layer}_rs{args.random_seed}{f"_downsample_{args.downsample}" if args.downsample < 1 else ""}_validation_results_over_training.json', 'w') as f:
    json.dump(results, f)

# Save probe
if args.save:
    torch.save(linear_probe, f'{args.path_to_reps}/saved_probes/{args.model_name.split("/")[-1]}_linear_probe_layer_{args.layer}_rs{args.random_seed}{f"_downsample_{args.downsample}" if args.downsample < 1 else ""}.pt')
