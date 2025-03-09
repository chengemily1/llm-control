"""Utilities for processing the Elix dataset."""

import pandas as pd
import numpy as np
from datasets import load_dataset
from scipy.optimize import minimize

def process_elix_dataset(user, dataset_path, exp_config=None, data_fraction=0.1):
    """Process the Elix dataset for a specific user level.
    
    Args:
        user (str): User type ('child', 'preteen', 'teenager', 'young adult', 'expert')
        dataset_path (str): Path to save the processed dataset
        exp_config (ExperimentConfig): Experiment configuration object
        data_fraction (float): Fraction of data to keep (default: 0.1)
    """
    # Define the users
    user_id = {'child': 1, 'preteen': 2, 'teenager': 3, 'young adult': 4, 'expert': 5}
    
    # Get random seed from config or use default
    random_seed = exp_config.random_seed if exp_config is not None else 42
    
    # Load dataset
    dataset = load_dataset("Asap7772/elix_generations_gpt4omini_pref")
    print("Dataset loaded successfully!")

    # Convert to DataFrame
    df = pd.DataFrame(dataset['train'])
    
    # Remove unwanted columns
    columns_to_drop = ['level_x', 'level_y', 'model_name_x', 'model_name_y', 'scorer_level', '__index_level_0__','det_choice']
    df = df.drop(columns=columns_to_drop)
    
    # Filter by scorer level
    df = df[df['scorer_level_id'] == user_id[user]]
    df = df.sample(n=len(df), random_state=random_seed)
    df = df.drop(columns='scorer_level_id')
    # Keep specified fraction of rows, randomly selected
    df = df.sample(n=int(len(df) * data_fraction), random_state=random_seed)
    
    # Calculate Bradley-Terry scores
    response_scores = calculate_bradley_terry_scores(df)
    print("\nBradley-Terry scores calculated successfully!")
    
    # Split into individual responses with scores
    final_df = split_comparisons_to_single_responses(df, response_scores, random_seed)
    
    # Save processed dataset
    final_df.to_csv(dataset_path, index=False)
    print(f"\nProcessed dataset saved to {dataset_path}")
    
    return final_df

def calculate_bradley_terry_scores(df):
    """Calculate scores for each response using the Bradley-Terry model."""
    # Create a mapping of unique responses to indices
    responses_x = df['response_x'].unique()
    responses_y = df['response_y'].unique()
    all_responses = np.unique(np.concatenate([responses_x, responses_y]))
    response_to_idx = {response: idx for idx, response in enumerate(all_responses)}
    
    # Initialize win matrix
    n_responses = len(all_responses)
    wins = np.zeros((n_responses, n_responses))
    
    # Fill win matrix
    for _, row in df.iterrows():
        i = response_to_idx[row['response_x']]
        j = response_to_idx[row['response_y']]
        if row['label'] == 0:  # response_x preferred
            wins[i, j] += 1
        else:  # response_y preferred
            wins[j, i] += 1
    
    # Function to compute negative log likelihood
    def neg_log_likelihood(params):
        # Add a regularization term to prevent extreme values
        reg_term = 0.1 * np.sum(params**2)
        
        # Compute log likelihood
        log_lik = 0
        for i in range(n_responses):
            for j in range(n_responses):
                if wins[i,j] > 0:
                    p_ij = np.exp(params[i]) / (np.exp(params[i]) + np.exp(params[j]))
                    log_lik += wins[i,j] * np.log(p_ij)
        
        return -(log_lik - reg_term)
    
    # Optimize to find best parameters
    initial_params = np.zeros(n_responses)
    result = minimize(neg_log_likelihood, initial_params, method='BFGS')
    
    # Convert parameters to probabilities and create response-score mapping
    scores = np.exp(result.x)
    scores = (scores - scores.min()) / (scores.max() - scores.min())  # normalize to [0,1]
    response_scores = {response: float(scores[idx]) for response, idx in response_to_idx.items()}
    
    return response_scores

def split_comparisons_to_single_responses(df, response_scores, random_seed):
    """Split comparison pairs into individual responses with their B-T scores.
    
    Args:
        df (pd.DataFrame): DataFrame with paired comparisons
        response_scores (dict): Dictionary mapping responses to their B-T scores
        random_seed (int): Random seed for reproducibility
    
    Returns:
        pd.DataFrame: DataFrame with individual responses and their scores
    """
    # Create separate rows for response_x
    df_x = pd.DataFrame({
        'prompt': df['prompt'],
        'response': df['response_x'],
        'level': df['level_id_x'],
        'score': df['response_x'].map(response_scores)
    })
    
    # Create separate rows for response_y
    df_y = pd.DataFrame({
        'prompt': df['prompt'],
        'response': df['response_y'],
        'level': df['level_id_y'],
        'score': df['response_y'].map(response_scores)
    })
    
    # Combine both sets of responses
    combined_df = pd.concat([df_x, df_y], ignore_index=True)
    
    # Remove duplicates (same response might appear in multiple pairs)
    combined_df = combined_df.drop_duplicates()
    
    # Create prompt_response field
    combined_df['prompt_response'] = combined_df['prompt'] + ' ' + combined_df['response']
    
    # Sort by score for easier inspection
    combined_df = combined_df.sort_values('score', ascending=False)
    
    # Print level-wise statistics
    print("\nScore statistics by level:")
    level_stats = combined_df.groupby('level').agg({
        'score': ['count', 'mean', 'std', 'min', 'max']
    })
    print("\nDetailed statistics by level:")
    print(level_stats)
    
    # Print average scores by level, sorted from highest to lowest
    print("\nAverage scores by level (sorted):")
    avg_scores = combined_df.groupby('level')['score'].mean().sort_values(ascending=False)
    for level, score in avg_scores.items():
        print(f"Level {level}: {score:.3f}")
    
    # Shuffle the dataset
    combined_df = combined_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    return combined_df

def save_processed_dataset(df, path, random_seed):
    """Save the processed dataset."""
    # Shuffle and save
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"\nReorganized dataset saved to {path}")

    # Print statistics
    print("\nReorganized dataset columns:")
    print(df.columns.tolist())
    
    print("\nNumber of rows for each score:")
    score_counts = df.groupby('score').size().sort_index()
    for score, count in score_counts.items():
        print(f"Score {score:.3f}: {count} rows") 