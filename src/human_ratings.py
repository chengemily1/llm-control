import json
import pandas as pd
import numpy as np
import torch
import argparse
import json
from transformers import pipeline
import pdb
import random
from utils.results_utils import load_generations

def parse_args():
    parser = argparse.ArgumentParser(description='training proof-of-concept')

    # Data selection
    parser.add_argument('--experiment', type=str, default='toxicity')
    parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument('--method', default='act', choices=['baseline', 'ours', 'act', 'actadd', 'instruct', 'fudge'])
    parser.add_argument('--liseco_lower', type=float, default=0.0)
    parser.add_argument('--liseco_upper', type=float, default=0.3)
    parser.add_argument('--liseco_map', default='sigmoid', choices=['sigmoid', 'tanh', 'identity'])
    parser.add_argument('--c', help="Actadd intervention strength", type=float, default=3)
    parser.add_argument('--l', default=6, type=int)
    parser.add_argument('--s', default=None, type=float)
    parser.add_argument('--config', default='config.json', help='path to config file')
    parser.add_argument('--downsample', default=None, type=float, help='downsample the dataset')
    args = parser.parse_args()

    return args

def terminal_ui(generations):
    print('Welcome to the human ratings interface!')
    print('Please rate the following generations on a scale of 1 to 5.')
    print('1 is completely unnatural, 5 is completely natural.')
    print('Press enter to continue...')
    input()

    samples = random.sample(generations, 50)
    ratings = []
    for j, gen in enumerate(samples):
        print(f'{j+1}/{len(samples)}: {gen}')
        rating = None
        while rating is None:
            rating = input('Rating: ')
            try:
                rating = int(rating)
            except ValueError:
                print('Invalid rating. Please enter a number between 1 and 5.')
                rating = None
        ratings.append(rating)
        print('---\n')

    return samples, ratings


if __name__ == "__main__":
    args = parse_args()
    print(args)

    # Load the dataset
    _, generations_df = load_generations(args)

    results = {}
    for setting in generations_df['strength'].unique():
        setting_df = generations_df[generations_df['strength']==setting]
        generations = list(setting_df['generation'])

        samples, ratings = terminal_ui(generations)
        mean_rating = np.mean(ratings)
        std_rating = np.std(ratings)
        results[setting] = {'mean': mean_rating, 'std': std_rating}

    print('Overall results: ', results)

    # Save the ratings
    with open(f'{args.method}_human_ratings.json', 'w') as f:
        json.dump(results, f)
