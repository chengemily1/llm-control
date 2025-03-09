import pandas as pd
import json
import os
import argparse
import numpy as np

# Dictionary mapping user types to their full descriptions
USER_DESCRIPTIONS = {
    "child": "child in elementary school",
    "preteen": "preteen in middle school",
    "teenager": "teenager in high school",
    "young adult": "young adult in college",
    "expert": "expert in the field"
}

def convert_csv_to_json(baseline_csv: str, controlled_csv: str, output_dir: str, user_type: str, seed: int):
    """
    Convert CSV files to separate JSON files for each model, preserving model identity
    """
    # Get full user description from dictionary or use the raw input if not found
    user_description = USER_DESCRIPTIONS.get(user_type, user_type)
    
    # Read CSV files
    baseline_df = pd.read_csv(baseline_csv)
    controlled_df = pd.read_csv(controlled_csv)
    
    # Create separate results for each model
    baseline_results = {}
    controlled_results = {}
    
    for idx in range(len(baseline_df)):
        question = baseline_df.iloc[idx]["question"]
        baseline_results[f"q{idx}"] = {
            "question": question,
            "response": baseline_df.iloc[idx]["generated_text"],
            "model": "baseline",
            "user_description": "The user is a " + USER_DESCRIPTIONS[args.user]
        }
        controlled_results[f"q{idx}"] = {
            "question": question,
            "response": controlled_df.iloc[idx]["generated_text"],
            "model": "controlled",
            "user_description": "The user is a " + USER_DESCRIPTIONS[args.user]
        }
    
    # Save separate JSON files
    os.makedirs(output_dir, exist_ok=True)
    
    baseline_path = os.path.join(output_dir, "llama-3-8b-baseline-results.json")
    controlled_path = os.path.join(output_dir, "llama-3-8b-controlled-results.json")
    
    with open(baseline_path, "w") as f:
        json.dump(baseline_results, f, indent=2)
    
    with open(controlled_path, "w") as f:
        json.dump(controlled_results, f, indent=2)
    
    print(f"Conversion completed! JSON files have been saved as:")
    print(f"- Baseline: {baseline_path}")
    print(f"- Controlled: {controlled_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert CSV files to evaluation JSON format')
    parser.add_argument('--user', type=str, required=True,
                      help='User type (e.g., child, preteen, teenager, young adult, expert)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    args = parser.parse_args()
    
    baseline_csv = "/scr/biggest/carmen/llm-control/experiments/elix_old/control_results/Meta-Llama-3-8B_baseline.csv"
    controlled_csv = "/scr/biggest/carmen/llm-control/experiments/elix_old/control_results/Meta-Llama-3-8B_ours.csv"
    output_dir = "results"
    
    convert_csv_to_json(baseline_csv, controlled_csv, output_dir, args.user, args.seed) 