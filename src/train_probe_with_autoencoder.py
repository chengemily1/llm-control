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

from autoencoder import AutoencoderWithLogisticRegression

parser = argparse.ArgumentParser(description='Train probe')

# Data selection
parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3-8B")
parser.add_argument('--experiment', type=str, choices=['toxicity', 'sentiment', 'formality'])
parser.add_argument('--objective', type=str, default='classification', choices=['regression', 'classification'])
# parser.add_argument('--dataset_name', type=str, default='/home/echeng/llm-control/jigsaw-toxic-comment-classification-challenge')
parser.add_argument('--device', type=str, default='cuda')
# parser.add_argument('--path_to_reps', type=str, default='/home/echeng/llm-control/experiments/toxicity/')
parser.add_argument('--num_epochs', type=int, default=10000)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--layer', type=int)
parser.add_argument('--random_seed', type=int)
parser.add_argument('--downsample', type=float, default=0.1)
parser.add_argument('--save', type=int, choices=[0,1], default=0, help='whether to save the probe')
parser.add_argument('--config', type=str, help='src/config.json')

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
    label = 'negative'
elif args.experiment == 'toxicity':
    label = 'toxic'
elif args.experiment == 'formality':
    label = 'formal'
    cont_label = 'avg_score'

def get_train_val(layer, split_percent=0.8):
    """
    Split representations into training and validation set.
    """
    labels = list(dataset[label]) # Get the binary labels
    labels = [int(label) if type(label)==bool else label for label in labels]

    if args.objective == 'classification':
        toxic = [i for i in range(len(labels)) if labels[i]]
        untoxic = random.sample([i for i in range(len(labels)) if not labels[i]], len(toxic))
    elif args.objective == 'regression':
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
    else:
        # TODO
        train_labels = [data[train_feat] for train_feat in train_features]
        # train_labels = [[data[train_feat], 1 - data[train_feat]] for train_feat in train_features]

    val_features = [elt for elt in data if elt not in set(train_features)]
    if args.objective == 'classification':
        val_labels = [data[val_feat] for val_feat in val_features]
    else:
        val_labels = [data[val_feat] for val_feat in val_features]

        # val_labels = [[data[val_feat], 1 - data[val_feat]] for val_feat in val_features] # p(formal) or p(toxic), 1-p

    train_features = torch.stack(train_features).float().to(device)
    val_features = torch.stack(val_features).float().to(device)

    train_labels = torch.tensor(train_labels, device=device)
    val_labels = torch.tensor(val_labels, device=device)

    return train_features, train_labels, val_features, val_labels

# Load representations
fpath = args.path_to_reps + f'saved_reps/{args.model_name.split("/")[-1]}_reps.pt'
representations = torch.load(fpath)[args.layer]

#### TRAIN
num_epochs = args.num_epochs
learning_rate = 0.0001

# Iterate over layers
train_features, train_labels, val_features, val_labels = get_train_val(representations)

if args.downsample < 1:
    train_features = train_features[:int(len(train_features) * args.downsample)]
    train_labels = train_labels[:int(len(train_labels) * args.downsample)]

# Define linear probe
pretrained_model_output_dim = representations[10].shape[-1] # take a random layer e.g. layer 10 and get the output dim
autoencoder = AutoencoderWithLogisticRegression(pretrained_model_output_dim, 10000, args.objective)

# Define optimizer
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)

accs, f1s, r2s, losses = [], [], [], []

# if args.objective == 'classification':
train_labels = train_labels.float().unsqueeze(-1)
val_labels = val_labels.float().unsqueeze(-1)

# Training loop
early_stop = False
best_loss = float('inf')

for epoch in range(num_epochs):
    autoencoder.train()
    optimizer.zero_grad()
    x_recon, y_pred = autoencoder(train_features)

    # Compute loss
    recon_loss = autoencoder.reconstruction_loss(train_features, x_recon)
    task_loss = autoencoder.task_loss(y_pred, train_labels)
    total_loss = autoencoder.tradeoff * recon_loss + (1 - autoencoder.tradeoff) * task_loss

    # Backward pass
    total_loss.backward()
    optimizer.step()

    # Validation
    autoencoder.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        x_recon, y_pred = autoencoder(val_features)

        recon_loss = autoencoder.reconstruction_loss(val_features, x_recon)
        task_loss = autoencoder.task_loss(y_pred, val_labels)
        total_loss = autoencoder.tradeoff * recon_loss + (1 - autoencoder.tradeoff) * task_loss

        if len(losses) > 1000:
            if total_loss > best_loss:
                early_stop = True
            else:
                best_loss = total_loss

        losses.append(val_loss)

        total = val_labels.size(0)

        if args.objective == 'classification':
            predicted = (nn.functional.sigmoid(y_pred) > 0.5).long().float() # convert to 0s and 1s
            # print('predicted vs val labels')
            # pdb.set_trace()
            correct = predicted.eq(val_labels).sum().item()
            binary_val = val_labels

            accuracy = 100. * correct / total
            # pdb.set_trace()
            f1 = f1_score(binary_val.cpu(), predicted.cpu())
            f1s.append(f1)
            accs.append(accuracy)

            print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Recon Loss: {recon_loss:.4f}, Task Loss: {task_loss:.4f}, Total Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%, F1: {f1:.2f}')

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
    torch.save(autoencoder, f'{args.path_to_reps}saved_probes/{args.model_name.split("/")[-1]}_linear_probe_layer_{args.layer}_rs{args.random_seed}{f"_downsample_{args.downsample}" if args.downsample < 1 else ""}.pt')
