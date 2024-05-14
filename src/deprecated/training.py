import argparse
import torch 
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import pdb
import random
from sklearn.metrics import f1_score 
from scipy import optimize  


parser = argparse.ArgumentParser(description='training proof-of-concept')

# Data selection
parser.add_argument('--experiment_name', type=str, default='toxicity')
parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument('--dataset_name', type=str, default='/home/echeng/llm-control/jigsaw-toxic-comment-classification-challenge')
# parser.add_argument('--layer', type=list, default=26)
parser.add_argument('--layers', metavar='N', type=int, nargs='+',
                        help='an integer or a list of integers')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--gamma', type=float)
parser.add_argument('--norm', type=float)
parser.add_argument('--norm_factor', type=float)
parser.add_argument('--svm_thres', type=float, default=0.05)
args = parser.parse_args()

ACCESS_TOKEN='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'


# Linear control wrapper class
class LinearControlWrapper(torch.nn.Module):
    def __init__(self, base_layer: nn.Module, 
                 linear_probe: nn.Module, name="", gamma=None, thres=0.05, norm=None,
                 norm_factor=None):
        """
        W shape: d x 2
        """
        super(LinearControlWrapper, self).__init__()
        self.base_layer = base_layer
        self.thres = - thres # the SVM threshold we're comfortable with

        # Probe-related parameters
        self.gamma = gamma
        self.norm = norm
        self.norm_factor = norm_factor
        self.probe = linear_probe.eval().half()
        self.W = linear_probe.weight # linear probe
        self.w1 = self.W[0,:].detach().cpu().numpy()
        self.w2 = self.W[1,:].detach().cpu().numpy()
        self.b = linear_probe.bias
        self.w = self.w1 - self.w2 # as defined in algo w_1 - w_2
        self.w_norm = np.linalg.norm(self.w) # python float
        self.w2T_w = self.w2 @ self.w

        # Logging
        self.toxic_sequences = []
        self.pre_adjust_toxicity_log = []
        self.post_adjust_toxicity_log = []

        # Control off or on
        self.control = False

    def control_off(self):
        self.control = False

    def control_on(self):
        self.control = True

    def eval_probe(self, x_seq, last_token_idx):
        eval_result = self.probe(x_seq[torch.arange(x_seq.size(0)),last_token_idx])
        return nn.functional.softmax(eval_result, dim=-1)[:,1].detach().cpu().numpy() # this is the probscore

    def forward(self, x, *args, **kwargs):
        x_seq, x_metadata = self.base_layer(x, *args, **kwargs)

        # Add the toxicity score to the log
        last_token_idx = kwargs['position_ids'].cpu().size(1) - 1
        self.pre_adjust_toxicity_log.append(self.eval_probe(x_seq, last_token_idx))
        
        if self.control: # Make the adjustment
            print('NORM OF LAST TOKEN: ', torch.norm(x_seq[torch.arange(x_seq.size(0)),last_token_idx], p=2))
            x_seq[torch.arange(x_seq.size(0)),last_token_idx] += self.optimal_theta(
                x_seq[torch.arange(x_seq.size(0)),last_token_idx] # get last token rep
            )

        # Add to toxicity log
        self.post_adjust_toxicity_log.append(self.eval_probe(x_seq, last_token_idx)) # this is the probscore

        return x_seq, x_metadata

    def optimal_theta(self, x):
        """Finds the optimal steering vector.

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        theta = torch.zeros(x.shape) # batch size x d
        x = x.detach().cpu().numpy()

        # Classified as toxic when (w_1 - w_2).T x < 0
        toxic_sequences_idx = np.where(x @ self.w < self.thres)
        self.toxic_sequences.append(toxic_sequences_idx)

        if not len(toxic_sequences_idx[0]): 
            print('no intervention needed')
            return theta.to(self.W.device)

        print('intervention needed')
        print(f"Sequence is toxic at token {len(self.pre_adjust_toxicity_log)}")

        x_toxic = x[toxic_sequences_idx] # index into the toxic ones only
        
        if self.gamma is not None:
            lmbda = self.convex_optimization_procedure(len(toxic_sequences_idx[0]), x_toxic)

            assert lmbda.success == True 
            print('VECTOR NORM: ', np.linalg.norm(lmbda.x * self.w))
            pdb.set_trace()
            theta[toxic_sequences_idx] = torch.Tensor(
                np.expand_dims(lmbda.x, axis=1) @ np.expand_dims(self.w, axis=0)
            )
        elif self.norm_factor is not None:
            theta[toxic_sequences_idx] = np.linalg.norm(x) * self.norm_factor / self.w_norm * torch.Tensor(self.w).unsqueeze(0)
        elif self.norm is not None:
            pdb.set_trace()
            theta[toxic_sequences_idx] = self.norm / self.w_norm * torch.Tensor(self.w).unsqueeze(0)
        return theta.to(self.W.device)
    
    def convex_optimization_procedure(self, size, x_toxic):
        root_function = lambda l: l * np.exp(l * self.w_norm**2 + x_toxic @ self.w) + l - 1/self.gamma
        
        # Binary search for initial condition
        x0 = np.ones(size,) * min(0.01, 0.25 * 1/self.gamma) 
        while np.all(root_function(x0) > 0):
            x0 = x0 * 0.5 # we know the positive root is between 0 and 1/gamma
        x0 = 2 * x0

        lmbda = optimize.root(root_function, x0, tol=1e-6) # parameter to optimize
        print(lmbda)

        assert lmbda.success == True 
        print('VECTOR NORM: ', np.linalg.norm(lmbda.x * self.w))
        pdb.set_trace()
        return lmbda

    def reset_logs(self):
        self.toxic_sequences = []
        self.pre_adjust_toxicity_log = []
        self.post_adjust_toxicity_log = []


# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=ACCESS_TOKEN)
model = AutoModelForCausalLM.from_pretrained(args.model_name, 
                                             token=ACCESS_TOKEN,
                                             load_in_8bit=True
                                            )

if 'Llama-2' in args.model_name:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

model.eval()

# Load all linear probes
Ws = [
    torch.load(
        f'/home/echeng/llm-control/experiments/{args.experiment_name}/saved_probes/{args.model_name.split("/")[-1]}_linear_probe_layer_{layer}.pt'
        ).to(args.device)
    for layer in range(len(model.model.layers))
]
[W.eval() for W in Ws]

# Load the dataset
dataset = pd.read_csv(args.dataset_name + '/test.csv').sample(frac=0.001)
data = ['Monica Lewinsky is such']#, 'monica lewinsky is such a', 'asian people are'] 
# data = list(dataset['comment_text'])[:5]

def encode_data(tokenizer, N, data, batch_size, max_length, device, last_k=None):
    # If the input data is text
    if type(data[0]) == str:
        encodings = tokenizer(data, padding=True, truncation=True, max_length=max_length, return_length=True, return_tensors="pt") # output variable length encodings
        if not last_k:
            encodings = [
                {'input_ids': encodings['input_ids'][i: i + batch_size].to(device),
                'attention_mask': encodings['attention_mask'][i: i + batch_size].to(device),
                'length': encodings['length'][i: i + batch_size] }
                for i in range(0, N, batch_size)
            ]
        else:
            encodings = [
                {'input_ids': encodings['input_ids'][i: i + batch_size][-last_k:].to(device),
                'attention_mask': encodings['attention_mask'][i: i + batch_size][-last_k:].to(device) }
                for i in range(0, N, batch_size)
            ]
    else: # input data is tokens-- manually pad and batch.
        max_len = max([len(sentence) for sentence in data])
        data = [sentence for sentence in data if len(sentence) > 2]
        encodings = [tokenizer.encode(sentence[1:], padding='max_length', max_length=max_len, return_tensors="pt") \
                     for sentence in data]
        batched_encodings = [torch.stack(encodings[i: i + batch_size]).squeeze(1).to(device) for i in range(0, len(data), batch_size)]
        batched_attention_masks = [(tokens != 1).to(device).long() for tokens in batched_encodings]
        encodings = [
            {'input_ids': batched_encodings[j], 'attention_mask': batched_attention_masks[j]}
            for j in range(len(batched_encodings))
        ]

    return encodings

encodings = [
    encode_data(tokenizer, 1, [datum], args.batch_size, model.config.max_position_embeddings, args.device)[0]
    for datum in data
]

# ORIGINAL BASELINE
model.eval()
print(args)
pdb.set_trace()

def retrofit_model(model, Ws):
    # Wrap all of the layers of the model
    for layer in range(len(model.model.layers)):
        model.model.layers[layer] = LinearControlWrapper(
            model.model.layers[layer], 
            linear_probe=Ws[layer], 
            gamma=args.gamma, 
            norm=args.norm, 
            norm_factor=args.norm_factor
        )

retrofit_model()
results_dict = {}

for i, encoding in enumerate(encodings):
    for layer in args.layers:
        model.model.layers[layer] = LinearControlWrapper(
            model.model.layers[layer].base_layer, 
            linear_probe=Ws[layer], 
            gamma=args.gamma, 
            norm=args.norm, 
            norm_factor=args.norm_factor
        )
        results_dict[layer] = {}
        results_dict[layer][data[i]] = {
            'baseline': {}, 'ours': {}
        }

    for method in ('baseline', 'ours'):
        if method == 'ours': 
            for layer in args.layers:
                model.model.layers[args.layer].control_on()
        elif method == 'baseline':
            for layer in range(len(model.model.layers)):
                model.model.layers[layer].control_off()
        for layer in range(len(model.model.layers)):
            model.model.layers[layer].reset_logs()
    
        # Generate output
        outputs = model.generate(
            inputs=encoding['input_ids'], # batch size x seq len 
            max_new_tokens=3,
            do_sample=False,
            return_dict_in_generate=True, 
            output_scores=True
        )
        generated_tokens = outputs.sequences
        transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        
        # Surface forms log: the text generated + surprisal
        generated_tokens_text = [tokenizer.decode(token) for token in generated_tokens[0]]
        surprisals = [-float(score.cpu().numpy()) for score in transition_scores[0]]
        results_dict[data[i]][method]['token'] = generated_tokens_text
        results_dict[data[i]][method]['surprisal'] = surprisals
        
        # Semantic scores log: 
        for layer in range(len(model.model.layers)):
            pre_adjust_scores = [float(score.item()) for score in model.model.layers[layer].pre_adjust_toxicity_log.copy()]
            post_adjust_scores = [float(score.item()) for score in model.model.layers[layer].post_adjust_toxicity_log.copy()]
            results_dict[layer][data[i]][method]['pre_adjust_toxicity_prob'] = pre_adjust_scores
            results_dict[layer][data[i]][method]['post_adjust_toxicity_prob'] = post_adjust_scores

# Save data
print(results_dict)
# with open(f'/home/echeng/llm-control/experiments/{args.experiment_name}/control_results/{args.model_name.replace("/", "_")}_gamma_{args.gamma}.json', 'w') as f:
#     json.dump(results_dict, f)
