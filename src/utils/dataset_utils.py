"""Utilities for processing the Elix dataset."""

import pandas as pd
import numpy as np
from datasets import load_dataset
from scipy.optimize import minimize
from time import time
import os
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt
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
    # df = df.sample(n=int(len(df) * data_fraction), random_state=random_seed)
    # Keep specified fraction of unique questions
    unique_questions = df['prompt'].unique()
    print(f"Number of unique questions: {len(unique_questions)}")
    df = df[df['prompt'].isin(unique_questions[:int(len(unique_questions) * data_fraction)])]
    print(f"After filtering to {data_fraction} of unique questions: {len(df)} rows remaining")

    # JL 3/9/25: sanity check. Choose one question and filter to only that question
    """
    prompt = df['prompt'].unique()[1]
    df = df[df['prompt'] == prompt]
    print(f"Sanity check: {len(df)} rows remaining")
    # Save sanity check to file
    sanity_check_path = os.path.join(os.path.dirname(dataset_path), f"sanity_check_{user}.csv")
    os.makedirs(os.path.dirname(sanity_check_path), exist_ok=True)
    df.to_csv(sanity_check_path, index=False)
    print(f"Sanity check saved to {sanity_check_path}")
    """
    
    # Calculate Bradley-Terry scores
    # response_scores, _ = calculate_bradley_terry_scores(df)
    response_scores = calculate_bradley_terry_per_question(df)
    print("\nBradley-Terry scores calculated successfully!")
    
    # Split into individual responses with scores
    final_df = split_comparisons_to_single_responses(df, response_scores, random_seed)

    # Plot histogram of scores
    plt.hist(final_df['score'], bins=20)
    plt.savefig(os.path.join(os.path.dirname(dataset_path), f"score_histogram.png"))
    
    # Save processed dataset
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    final_df.to_csv(dataset_path, index=False)
    print(f"\nProcessed dataset saved to {dataset_path}")
    
    return final_df

def get_empirical_probabilities(wins):
    # Get total matches for each pair
    total_matches = wins + wins.T
    
    # Compute empirical win probabilities (add small constant to avoid division by zero)
    eps = 1e-8
    empirical_probs = wins / (total_matches + eps)
    
    # Average the win probabilities for each team
    avg_win_prob = empirical_probs.mean(axis=1)
    
    # Convert to log space for initial parameters (add eps to avoid log(0))
    initial_params = np.log(avg_win_prob + eps)
    
    # Center the parameters (helps with numerical stability)
    initial_params = initial_params - initial_params.mean()
    
    return initial_params

def calculate_bradley_terry_scores_torch(wins, n_epochs=1000, lr=0.01, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Convert wins matrix to torch tensor
    wins = torch.tensor(wins, dtype=torch.float32, device=device)
    n_responses = wins.shape[0]
    
    # Get smart initialization
    initial_params = get_empirical_probabilities(wins.cpu().numpy())
    print(f"Initial parameters: {initial_params}")
    params = torch.tensor(initial_params, requires_grad=True, device=device)
    
    # Use Adam with weight decay (L2 regularization)
    optimizer = torch.optim.Adam([params], lr=lr, weight_decay=0.1)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=100, verbose=True
    )
    
    def neg_log_likelihood():
        # Convert params to exp(params)
        exp_params = torch.exp(params)
        
        # Reshape for broadcasting
        exp_params_i = exp_params.unsqueeze(1)
        exp_params_j = exp_params.unsqueeze(0)
        
        # Compute probabilities
        p_ij = exp_params_i / (exp_params_i + exp_params_j)
        
        # Compute log likelihood (add small epsilon to avoid log(0))
        eps = torch.finfo(torch.float32).eps
        log_lik = torch.sum(wins * torch.log(p_ij + eps * (wins == 0)))
        
        return -log_lik  # Removed explicit regularization since using weight_decay
    
    # Training loop
    print("Starting optimization...")
    start_time = time()
    best_loss = float('inf')
    best_params = None
    patience = 0
    max_patience = 300
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = neg_log_likelihood()
        loss.backward()
        optimizer.step()
        
        # Learning rate scheduling
        scheduler.step(loss)
        
        # Save best parameters
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = params.detach().clone()
            patience = 0
        else:
            patience += 1
        
        # Early stopping
        if patience >= max_patience:
            print(f"Early stopping at epoch {epoch}")
            break
            
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    training_time = time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Return best parameters as numpy array
    return best_params.cpu().numpy()

def calculate_bradley_terry_scores(df):
    """Calculate scores for each response using the Bradley-Terry model."""
    # Create a mapping of unique responses to indices
    responses_x = df['response_x'].unique()
    responses_y = df['response_y'].unique()
    all_responses = np.unique(np.concatenate([responses_x, responses_y]))
    response_to_idx = {response: idx for idx, response in enumerate(all_responses)}
    
    # Initialize win matrix
    n_responses = len(all_responses)
    print(f"Number of responses: {n_responses}")
    wins = np.zeros((n_responses, n_responses))
    
    # Fill win matrix
    for _, row in df.iterrows():
        i = response_to_idx[row['response_x']]
        j = response_to_idx[row['response_y']]
        if row['label'] == 0:  # response_x preferred
            wins[i, j] += 1
        else:  # response_y preferred
            wins[j, i] += 1

    # Use the function
    result = calculate_bradley_terry_scores_torch(wins)
    
    # Convert parameters to probabilities and create response-score mapping
    # scores = np.exp(result.x)
    scores = result
    scores = (scores - scores.min()) / (scores.max() - scores.min())  # normalize to [0,1]
    response_scores = {response: float(scores[idx]) for response, idx in response_to_idx.items()}
    
    return response_scores, wins


def calculate_bradley_terry_per_question(df):
    """Calculate scores for each response using the Bradley-Terry model for each question."""
    # Get unique questions
    unique_questions = df['prompt'].unique()
    print(f"Number of unique questions: {len(unique_questions)}")
    full_response_scores = {} # assumes there are no responses that correspond to multiple questions

    for question in unique_questions:
        df_question = df[df['prompt'] == question]
        response_scores, _ = calculate_bradley_terry_scores(df_question)
        full_response_scores.update(response_scores)

    print(f"Number of responses: {len(full_response_scores.keys())}")
    return full_response_scores


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