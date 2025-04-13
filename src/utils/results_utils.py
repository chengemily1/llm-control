import os
import json
import pandas as pd

with open('config.json', 'r') as f:
    config = json.load(f)

def get_results_json_path(args):
    this_path = config['path']
    model_name = args.model_name.split('/')[-1]
    if args.downsample is not None:
        downsample = f'_downsample_{args.downsample}'
    else:
        downsample = ''
    results_path = f'{this_path}/experiments/{args.experiment}/control_results/{model_name}_low_{args.liseco_lower}_high_{args.liseco_upper}_{args.method}{downsample}.json'
    return results_path

def load_results_json(args):
    """
    Loads the raw results json file for a given experiment.
    """
    results_path = get_results_json_path(args)
    with open(results_path, 'r') as f:
        results = json.load(f)

    return results

def load_results_csv(args):
    """
    Loads the csv file
    """
    this_path = config['path']

    model_name = args.model_name.split('/')[-1]
    results_path = f'{this_path}/experiments/{args.experiment}/control_results/{model_name}_linear_act.csv'

    return pd.read_csv(results_path)

def load_generations(args):
    """
        Load the generations from te dataset
    """
    if args.method == 'act':
        generations = load_results_csv(args)
        generation_list = list(generations['generation'])
    else:
        generations = load_results_json(args)
        generation_list = [generations[gen]['generated_text'] for gen in generations]
    return generation_list, generations
