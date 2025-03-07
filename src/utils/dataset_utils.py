"""Utilities for processing the Elix dataset."""

import pandas as pd
from datasets import load_dataset
import numpy as np
from collections import defaultdict

def process_elix_dataset(user, dataset_path):
    """Process the Elix dataset for a specific user level."""
    # Define the users
    user_id = {'child': 1, 'preteen': 2, 'teenager': 3, 'young adult': 4, 'expert': 5}
    
    # Load dataset
    dataset = load_dataset("Asap7772/elix_generations_gpt4omini_pref")
    print("Dataset loaded successfully!")

    # Convert to DataFrame
    df = pd.DataFrame(dataset['train'])
    
    # Remove unwanted columns
    columns_to_drop = ['level_x', 'level_y', 'model_name_x', 'model_name_y', 'scorer_level', '__index_level_0__']
    df = df.drop(columns=columns_to_drop)
    
    # Filter by scorer level
    df = df[df['scorer_level_id'] == user_id[user]]
    df = df.drop(columns='scorer_level_id')
    
    # Process and balance dataset
    level_scores = calculate_level_scores(df)
    print_level_scores(level_scores)
    reorganized_df = process_and_balance_dataset(df, level_scores)
    
    # Save processed dataset
    save_processed_dataset(reorganized_df, dataset_path)
    return reorganized_df

def fit_bradley_terry(df, max_iter=100, tol=1e-8):
    """Fit Bradley-Terry model to pairwise comparison data.
    
    Args:
        df: DataFrame with columns level_id_x, level_id_y, and label
        max_iter: Maximum number of iterations for MLE
        tol: Convergence tolerance
        
    Returns:
        dict: Mapping from level_id to strength parameter
    """
    # Get unique levels
    levels = set(df['level_id_x'].unique()) | set(df['level_id_y'].unique())
    n_levels = len(levels)
    level_to_idx = {level: idx for idx, level in enumerate(sorted(levels))}
    
    # Initialize win matrix
    W = np.zeros((n_levels, n_levels))
    N = np.zeros((n_levels, n_levels))
    
    # Fill matrices
    for _, row in df.iterrows():
        i, j = level_to_idx[row['level_id_x']], level_to_idx[row['level_id_y']]
        if row['label'] == 0:  # x preferred
            W[i, j] += 1
        else:  # y preferred
            W[j, i] += 1
        N[i, j] += 1
        N[j, i] += 1
    
    # Initialize strength parameters
    gamma = np.ones(n_levels) / n_levels
    
    # Iterative algorithm for MLE
    for _ in range(max_iter):
        gamma_old = gamma.copy()
        
        # Update each gamma[i]
        for i in range(n_levels):
            # Skip if no comparisons
            if np.sum(N[i, :]) == 0:
                continue
                
            # Calculate numerator (total wins)
            numerator = np.sum(W[i, :])
            
            # Calculate denominator
            denominator = 0
            for j in range(n_levels):
                if i != j and N[i, j] > 0:
                    denominator += N[i, j] * gamma[j] / (gamma[i] + gamma[j])
            
            # Update gamma[i]
            if denominator > 0:
                gamma[i] = numerator / denominator
        
        # Normalize to prevent numerical issues
        gamma = gamma / np.sum(gamma)
        
        # Check convergence
        if np.max(np.abs(gamma - gamma_old)) < tol:
            break
    
    # Convert back to probabilities and create mapping
    probs = gamma / np.sum(gamma)
    return {level: float(probs[idx]) for level, idx in level_to_idx.items()}

def calculate_level_scores(df):
    """Calculate scores for each level using Bradley-Terry model."""
    # Fit Bradley-Terry model
    level_scores = fit_bradley_terry(df)
    
    # Print comparison of scores
    print("\nBradley-Terry model scores:")
    for level, score in sorted(level_scores.items()):
        print(f"Level {level}: {score:.3f}")
    
    return level_scores

def print_level_scores(level_scores):
    """Print the scores for each level."""
    print("\nLevel scores from Bradley-Terry model:")
    for level, score in sorted(level_scores.items()):
        print(f"Level {level}: {score:.3f}")

def process_and_balance_dataset(df, level_scores):
    """Process and balance the dataset."""
    # Split comparisons into separate rows
    reorganized_df = split_comparisons(df, level_scores)
    
    # Balance by score
    reorganized_df = balance_by_score(reorganized_df)
    
    print(f"\nOriginal dataset shape: {df.shape}")
    print(f"Reorganized dataset shape: {reorganized_df.shape}")
    
    return reorganized_df

def split_comparisons(df, level_scores):
    """Split comparison rows into separate response rows."""
    # Create DataFrame for response_x
    df_x = df.copy()
    df_x['response'] = df_x['response_x']
    df_x['level_id'] = df_x['level_id_x']
    df_x['score'] = df_x['level_id_x'].map(level_scores)
    df_x['prompt_response'] = df_x['prompt'] + ' ' + df_x['response_x']
    
    # Create DataFrame for response_y
    df_y = df.copy()
    df_y['response'] = df_y['response_y']
    df_y['level_id'] = df_y['level_id_y']
    df_y['score'] = df_y['level_id_y'].map(level_scores)
    df_y['prompt_response'] = df_y['prompt'] + ' ' + df_y['response_y']
    
    # Combine and clean up
    combined_df = pd.concat([df_x, df_y], ignore_index=True)
    
    # Keep only relevant columns
    columns_to_keep = ['prompt', 'response', 'prompt_response', 'level_id', 'score']
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