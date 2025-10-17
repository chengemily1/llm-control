import os
import json
import pandas as pd
import pdb

with open('/home/echeng/llm-control/src/config.json', 'r') as f:
    config = json.load(f)

def get_results_json_path(args):
    this_path = config['path']
    model_name = args.model_name.split('/')[-1]
    if args.downsample is not None and args.downsample < 1:
        downsample = f'_downsample_{args.downsample}'
    else:
        downsample = ''

    if args.method == 'actadd':
        results_path = f'{this_path}/experiments/{args.experiment}/control_results/{model_name}_{args.method}_c{args.c}_l{args.l}{downsample}.json'
    elif args.method == 'ours':
        results_path = f'{this_path}/experiments/{args.experiment}/control_results/{model_name}_{args.method}_lower_{args.liseco_lower}_upper_{args.liseco_upper}_map_{args.liseco_map}{downsample}.json'
    elif args.method == 'baseline':
        results_path = f'{this_path}/experiments/{args.experiment}/control_results/{model_name}_{args.method}{downsample}.json'
    elif args.method == 'instruct':
        results_path = f'{this_path}/experiments/{args.experiment}/control_results/{model_name}_instruct.json'

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
    results_path = f'{this_path}/experiments/{args.experiment}/control_results/act_{model_name}.csv'

    return pd.read_csv(results_path)

def load_generations(args):
    """
        Load the generations from te dataset
    """
    if args.method == 'act':
        generations = load_results_csv(args)
    else:
        generations = load_results_json(args)
        if type(generations) == list:
            print('here')
            df = []
            for generation in generations:
                df.append(["", generation['generated_text']]) # unconditional version
            generations = pd.DataFrame(df, columns=['prompt', 'generation'])
        elif type(generations) == dict:
            df = []
            for prompt in generations:
                df.append([prompt, generations[prompt]['generated_text']])
            generations = pd.DataFrame(df, columns=['prompt', 'generation'])
    return generations

def save_results_json(args, results):
    this_path = config['path']
    model_name = args.model_name.split('/')[-1]
    if args.method == 'act':
        prefix = args.method
    else:
        prefix = f'{args.method}_low_{args.liseco_lower}_high_{args.liseco_upper}'
    results_path = f'{this_path}/experiments/{args.experiment}/control_results/{model_name}_{prefix}_human_ratings.json'
    with open(results_path, 'w') as f:
        json.dump(results, f)

def save_results_csv(args, results):
    this_path = config['path']

    model_name = args.model_name.split('/')[-1]
    if args.method == 'ours':
        hyperparameters = f'low_{args.liseco_lower}_high_{args.liseco_upper}'
    elif args.method == 'actadd':
        hyperparameters = f'c_{args.c}_l_{args.l}'
    else:
        hyperparameters = f'{args.method}_'
    results_path = f'{this_path}/experiments/{args.experiment}/control_results/{model_name}_{hyperparameters}scores.csv'

    results.to_csv(results_path)
