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

random.seed(42)

parser = argparse.ArgumentParser(description='Train probe')

# Data selection
parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument('--dataset_name', type=str, default='/home/echeng/llm-control/jigsaw-toxic-comment-classification-challenge')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--path_to_reps', type=str, default='/home/echeng/llm-control/experiments/toxicity/')
parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--layer', type=int)
parser.add_argument('--save', type=int, choices=[0,1], default=0, help='whether to save the probe')

args = parser.parse_args()
print(args)

ACCESS_TOKEN='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'

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=ACCESS_TOKEN)
model = AutoModelForCausalLM.from_pretrained(args.model_name, 
                                             token=ACCESS_TOKEN,
                                             load_in_8bit=True
                                            )

# Idiosyncrasy of Llama 2
if 'Llama-2' in args.model_name:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

model.eval()
device = 'cuda:0' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
dataset = pd.read_csv(args.dataset_name + '/train_shuffled.csv') # we shuffled the ds in saving the reps

def get_train_val(layer, split_percent=0.8):
    """
    Split representations into training and validation set.
    """
    labels = list(dataset['toxic']) # Get the labels 
    toxic = [i for i in range(len(labels)) if labels[i]]
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

# Load representations
representations = torch.load(args.path_to_reps + 'reps.pt')[args.layer]

# Get linear probe
pretrained_model_output_dim = representations[10].shape[-1] # take a random layer e.g. layer 10 and get the output dim
num_classes = 2

# TRAIN
num_epochs = 1000
learning_rate = 0.0001

# Iterate over layers (rn it's doing every other layer)
train_features, train_labels, val_features, val_labels = get_train_val(representations)

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
        correct += predicted.eq(val_labels).sum().item()
    
    val_loss /= len(val_features)

    accuracy = 100. * correct / total
    f1 = f1_score(val_labels.cpu(), predicted.cpu())
    f1s.append(f1)
    accs.append(accuracy)
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%, F1: {f1:.2f}')

# Save validation accuracy, f1
results = {'val_acc': accs, 'val_f1': f1s, 'val_epoch': list(range(num_epochs))}
with open(args.path_to_reps + f'layer_{args.layer}_validation_results_over_training.json', 'w') as f:
    json.dump(results, f)

# Save probe
if args.save:
    torch.save(linear_probe, f'{args.path_to_reps}/linear_probe_layer_{args.layer}.pt')