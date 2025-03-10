"""Utilities for processing the GMS8K dataset."""

import pandas as pd
from datasets import load_dataset

def process_gms8k_dataset(dataset_path, exp_config=None):
    """Process the GMS8K dataset for a specific user level. """

    # Get random seed from config or use default
    random_seed = exp_config.random_seed if exp_config is not None else 42
    
    # Load dataset
    dataset = load_dataset("openai/gsm8k", "main")
    print("Dataset loaded successfully!")

    # Convert to DataFrame
    df = pd.DataFrame(dataset['train'])

    # Create combined text field
    df[exp_config.text_field] = df['question'] + ' ' + df['answer']
    
    # Extract numerical answers (assuming the answer contains a number)
    def extract_number(text):
        import re
        numbers = re.findall(r'[-+]?\d*\.?\d+', text)
        return float(numbers[-1]) if numbers else 0.0
    
    df['target'] = df['answer'].apply(extract_number)

    # Shuffle dataset
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Save shuffled dataset
    df.to_csv(dataset_path, index=False)
    print(f"\nReorganized dataset saved to {dataset_path}")
    