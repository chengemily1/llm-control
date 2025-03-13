"""Utilities for processing the reviews dataset."""

import pandas as pd
from datasets import load_dataset
import os

def process_reviews_dataset(user, dataset_path, exp_config=None, data_fraction=1.0):
    """Process the reviews dataset for a specific user.
    
    Args:
        user (str): User ID
        dataset_path (str): Path to save the processed dataset
        exp_config (ExperimentConfig): Experiment configuration object
        data_fraction (float): Fraction of data to keep (default: 1.0)
    """
    # Get random seed from config or use default
    random_seed = exp_config.random_seed if exp_config is not None else 42
    
    # Load dataset
    dataset = load_dataset("Asap7772/steered_reviews_full_autolabel_gpt4o_pref")
    print("Dataset loaded successfully!")

    # Convert to DataFrame
    df = pd.DataFrame(dataset['train'])
    
    # Remove unwanted columns if any
    columns_to_keep = ['prompt', 'level_x', 'level_y', 'response_x', 'response_y', 'label', 'scorer_level']
    df = df[columns_to_keep]
    
    # Filter by user
    df = df[df['scorer_level'] == user]
    print(f"Found {len(df)} examples for user {user}")
    
    # Process and balance dataset
    reorganized_df = process_and_balance_reviews(df)
    
    # Save processed dataset
    save_processed_dataset(reorganized_df, dataset_path)
    return reorganized_df

def process_and_balance_reviews(df):
    """Process and balance the reviews dataset."""
    # Split comparisons into separate rows
    reorganized_df = split_review_comparisons(df)
    
    # Balance by score
    reorganized_df = balance_by_score(reorganized_df)
    
    print(f"\nOriginal dataset shape: {df.shape}")
    print(f"Reorganized dataset shape: {reorganized_df.shape}")
    
    return reorganized_df

def split_review_comparisons(df):
    """Split comparison rows into separate response rows."""
    # Create DataFrame for response_x
    df_x = df.copy()
    df_x['response'] = df_x['response_x']
    df_x['score'] = 1 - df_x['label']  # If label=0, response_x was preferred (score=1)
    df_x['level'] = df_x['level_x']
    df_x['prompt_response'] = df_x['prompt'] + ' ' + df_x['response_x']
    
    # Create DataFrame for response_y
    df_y = df.copy()
    df_y['response'] = df_y['response_y']
    df_y['score'] = df_y['label']  # If label=1, response_y was preferred (score=1)
    df_y['level'] = df_y['level_y']
    df_y['prompt_response'] = df_y['prompt'] + ' ' + df_y['response_y']
    
    # Combine and clean up
    combined_df = pd.concat([df_x, df_y], ignore_index=True)
    
    # Keep only relevant columns
    columns_to_keep = ['prompt', 'response', 'prompt_response', 'scorer_level', 'score', 'level']
    combined_df = combined_df[columns_to_keep]
    
    # Remove duplicates
    combined_df = combined_df.drop_duplicates(subset=['prompt_response'])
    
    return combined_df

def balance_by_score(df):
    """Balance the dataset by score."""
    # Get the minimum count of any score
    score_counts = df.groupby('score').size()
    min_count = score_counts.min()
    
    # For each score, sample down to the minimum count
    balanced_dfs = []
    for score in df['score'].unique():
        score_df = df[df['score'] == score]
        if len(score_df) > min_count:
            balanced_df = score_df.sample(n=min_count, random_state=42)
        else:
            balanced_df = score_df
        balanced_dfs.append(balanced_df)
    
    # Combine all balanced dataframes
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # Print statistics
    print("\nBefore balancing:")
    print(df.groupby('score').size())
    print("\nAfter balancing:")
    print(balanced_df.groupby('score').size())
    
    return balanced_df

def save_processed_dataset(df, path):
    """Save the processed dataset."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Shuffle and save
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"\nReorganized dataset saved to {path}")

    # Print statistics
    print("\nReorganized dataset columns:")
    print(df.columns.tolist())
    
    print("\nNumber of rows for each score:")
    score_counts = df.groupby('score').size().sort_index()
    for score, count in score_counts.items():
        print(f"Score {score:.3f}: {count} rows") 