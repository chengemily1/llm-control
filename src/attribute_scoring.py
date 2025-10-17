import pandas as pd
import numpy as np
import torch
import argparse
import json
from transformers import pipeline
import pdb
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
from tqdm import tqdm

from utils.results_utils import load_generations, save_results_csv

def parse_args():
    parser = argparse.ArgumentParser(description='training proof-of-concept')

    # Data selection
    parser.add_argument('--experiment', type=str, default='toxicity')
    parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument('--method', default='ours', choices=['baseline', 'ours', 'act', 'actadd', 'instruct', 'fudge'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--liseco_lower', type=float, default=0.0)
    parser.add_argument('--liseco_upper', type=float, default=0.3)
    parser.add_argument('--liseco_map', default='sigmoid', choices=['sigmoid', 'tanh', 'identity'])
    parser.add_argument('--downsample', default=1.0, type=float)
    parser.add_argument('--c', help="Actadd intervention strength", type=float, default=3)
    parser.add_argument('--l', default=6, type=int)
    parser.add_argument('--s', default=None, type=float)
    parser.add_argument('--config', default='config.json', help='path to config file')
    args = parser.parse_args()

    return args


def get_scorer(args):
    """
        Load the HF pipeline for the given experiment
    """
    if args.experiment == 'toxicity':
        pp = pipeline(model="s-nlp/roberta_toxicity_classifier")
    elif args.experiment == 'sentiment':
        pp = pipeline("text-classification", model='cardiffnlp/twitter-roberta-base-sentiment-latest')
    elif args.experiment == 'formality':
        pp = pipeline("text-classification", model='s-nlp/roberta-base-formality-ranker')
    else:
        raise ValueError(f"Experiment {args.experiment} not supported")
    return pp

def score_text(args, generations: list[str], binary: bool = True):
    """
    Score the generations using the pipeline
    If binary is True, thresholds the scores at 0.5 to 0 or 1.
    Returns a list of scores p(toxic), one for each generation.
    """
    pp = get_scorer(args)

    # Score the generations
    scored_generations = pp(generations)

    # Get the forbidden label
    if args.experiment == 'sentiment':
        forbidden_label = 'negative'
    elif args.experiment == 'toxicity':
        forbidden_label = 'toxic'
    elif args.experiment == 'formality':
        forbidden_label = 'informal'
    else:
        raise ValueError(f"Experiment {args.experiment} not supported")

    scores = []
    for gen in scored_generations:
        score = gen['score'] if gen['label'] == forbidden_label else 1 - gen['score']
        if binary:
            score = 1 if score > 0.5 else 0
        scores.append(score)

    # memory management
    del pp

    return scores


def compute_perplexity(text, model, tokenizer, device):
    # Encode the input text
    encodings = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    perplexity = np.exp(loss.item())
    return perplexity

if __name__ == "__main__":
    args = parse_args()
    print(args)

    # Load the dataset
    generations_df = load_generations(args)
    generations = list(generations_df['generation'])

    # Load the pipeline / scorer
    print('scoring...')
    scores = score_text(args, generations, binary=False)

    generations_df['p_toxic'] = scores
    generations_df['toxic'] = generations_df['p_toxic'] > 0.5

    if args.method == 'act':
        print(generations_df.groupby('strength').agg({'p_toxic': [np.mean, np.std], 'toxic': [np.mean, np.std]}))
    else:
        print('Mean and std of p_toxic')
        print(np.mean(generations_df['p_toxic']), np.std(generations_df['p_toxic']))
        print('Mean and std of toxic')
        print(np.mean(generations_df['toxic']), np.std(generations_df['toxic']))
    
    # Score perplexity
    perplexities = []
    model_name = 'Qwen/Qwen2.5-3B'

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map='auto')
    model.eval()
    
    # Ensure model is on CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    for text in tqdm(generations):
        ppl = compute_perplexity(text, model, tokenizer, device)
        perplexities.append(ppl)
    
    print('mean ppl:', np.mean(perplexities))
    generations_df['ppl'] = perplexities

    # Save the scores
    save_results_csv(args, generations_df)



