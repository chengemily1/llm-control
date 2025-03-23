"""Utilities for processing the persona dataset."""

import pandas as pd
from datasets import load_dataset
import os
import logging
from typing import Dict, Any, List, Union
import ast
import torch
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_persona_string(persona_str: str) -> Dict[str, str]:
    """
    Parse a persona string into a dictionary of features.

    Args:
        persona_str (str): The persona string with key-value pairs separated by newlines

    Returns:
        Dict[str, str]: Dictionary of persona features
    """
    # Initialize an empty dictionary to store the persona features
    persona_dict = {}

    # Split the string by newlines
    lines = persona_str.strip().split("\n")

    # Process each line as a key-value pair
    for line in lines:
        if ":" in line:
            # Split by the first colon
            key, value = line.split(":", 1)
            # Clean and store the key-value pair
            persona_dict[key.strip()] = value.strip()

    return persona_dict


def extract_persona_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all features from persona strings and add them as columns to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing a 'persona' column with persona strings

    Returns:
        pd.DataFrame: DataFrame with added persona feature columns
    """
    logger.info("Extracting features from persona strings...")

    # Create a new DataFrame to store all persona features
    all_features = {}
    feature_keys = set()

    # Process each persona string
    for idx, persona_str in enumerate(df["persona"]):
        # Parse the persona string
        persona_dict = parse_persona_string(persona_str)

        # Store the features for this row
        all_features[idx] = persona_dict

        # Update the set of all feature keys
        feature_keys.update(persona_dict.keys())

    # Convert to DataFrame
    features_df = pd.DataFrame.from_dict(all_features, orient="index")

    # Add prefix to feature columns to avoid conflicts
    features_df = features_df.add_prefix("persona_")

    # Log the extracted features
    logger.info(
        f"Extracted {len(features_df.columns)} persona features: {features_df.columns.tolist()}"
    )

    # Concatenate with original DataFrame
    result_df = pd.concat([df.reset_index(drop=True), features_df], axis=1)

    # Fill in any "nan" with "N/A"
    result_df = result_df.fillna("N/A")

    return result_df


def extract_persona_representations(
    model,
    tokenizer,
    df: pd.DataFrame,
    output_dir: str,
    model_name: str,
    device: str = "cuda",
    layer_idx: int = -1,
    save_every: int = 10,
) -> Dict[int, Dict[str, Union[str, torch.Tensor]]]:
    """
    Extract representations for each persona feature using the model.

    Args:
        model: The language model
        tokenizer: The tokenizer for the model
        df (pd.DataFrame): DataFrame with persona feature columns
        output_dir (str): Directory to save the representations
        model_name (str): Name of the model (for filename)
        device (str): Device to use (default: "cuda")
        layer_idx (int): Layer to extract representation from (default: -1 for final layer)
        save_every (int): Save intermediate results every N personas (default: 10)

    Returns:
        Dict[int, Dict[str, Union[str, torch.Tensor]]]: Mapping from persona IDs to feature representations

        The output is a dictionary with the following structure:
        {
            persona_idx: {
                "key": key value as a string,
                ...
                "data": concatenated model embeddings for this persona
            }
        }
    """
    from src.utils.representation_utils import get_persona_representation

    logger.info(
        f"Extracting representations for persona features from layer {layer_idx}..."
    )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Dictionary to store all representations
    all_representations = {}

    # Ensure we have a persona_idx column
    if "persona_idx" not in df.columns:
        logger.warning("persona_idx column not found in DataFrame. Creating one.")
        # Create a mapping from unique persona strings to indices
        unique_personas = df["persona"].unique()
        persona_to_idx = {persona: idx for idx, persona in enumerate(unique_personas)}
        # Add persona_idx column to the DataFrame
        df["persona_idx"] = df["persona"].map(persona_to_idx)

    # Create a mapping from persona string to persona_idx
    persona_to_idx = dict(
        zip(
            df["persona"].unique(),
            df.loc[df["persona"].drop_duplicates().index, "persona_idx"],
        )
    )

    # Verify the mapping
    logger.info("Persona to index mapping:")
    for persona_str, idx in list(persona_to_idx.items())[:3]:
        logger.info(f"Persona (truncated) '{persona_str[:50]}...' -> idx {idx}")

    # Process each unique persona
    unique_personas = df["persona"].unique()
    logger.info(f"Processing {len(unique_personas)} unique personas...")

    for persona_str in tqdm(unique_personas, desc="Processing personas"):
        persona_idx = persona_to_idx[persona_str]

        # Parse the persona string
        persona_dict = parse_persona_string(persona_str)

        if int(persona_idx) == 0:
            logger.info(f"Persona {persona_idx} features: {persona_dict}")
            # Find a row with this persona
            sample_row = df[df["persona"] == persona_str].iloc[0]
            logger.info(f"Persona {persona_idx} data: {sample_row['data']}")
            logger.info(
                f"Persona {persona_idx} idx in DataFrame: {sample_row['persona_idx']}"
            )

        # Dictionary to store representations for this persona
        persona_reps = {}

        # List to store all feature embeddings for concatenation
        all_feature_embeddings = []

        # Process each feature
        for feature_key, feature_value in persona_dict.items():
            # Create a descriptive text for this feature
            feature_text = f"{feature_key}: {feature_value}"

            # Get representation for this feature from the specified layer
            rep = get_persona_representation(
                model, tokenizer, feature_text, device, layer_idx
            )

            # Store the representation
            persona_reps[feature_key] = feature_text

            # Add to list for concatenation
            all_feature_embeddings.append(rep)

        # Concatenate all feature embeddings
        if all_feature_embeddings:
            # Stack all embeddings along dimension 0
            concatenated_embedding = torch.cat(all_feature_embeddings, dim=0)

            # Add the concatenated embedding to the persona representations
            persona_reps["data"] = concatenated_embedding

        # Store all representations for this persona
        all_representations[int(persona_idx)] = persona_reps

        # Save intermediate results every save_every personas
        if len(all_representations) % save_every == 0:
            logger.info(
                f"Saving intermediate representations after {len(all_representations)} personas..."
            )
            save_persona_feature_representations(
                all_representations, output_dir, model_name, layer_idx
            )

    # Save final results
    save_persona_feature_representations(
        all_representations, output_dir, model_name, layer_idx
    )

    # Save a mapping file for reference
    mapping_path = os.path.join(
        output_dir, f"{model_name.split('/')[-1]}_persona_idx_mapping.txt"
    )
    with open(mapping_path, "w") as f:
        for persona_str, idx in persona_to_idx.items():
            f.write(f"Idx {idx}: {persona_str[:100]}...\n")
    logger.info(f"Saved persona_idx mapping to {mapping_path}")

    return all_representations


def save_persona_feature_representations(
    representations: Dict[str, Dict[str, torch.Tensor]],
    output_dir: str,
    model_name: str,
    layer_idx: int = -1,
):
    """
    Save persona feature representations to disk.

    Args:
        representations (Dict[str, Dict[str, torch.Tensor]]): Persona feature representations
        output_dir (str): Directory to save the representations
        model_name (str): Name of the model (for filename)
        layer_idx (int): Layer index used for extraction (default: -1 for final layer)
    """
    model_short_name = model_name.split("/")[-1]

    # Include layer information in the filename
    layer_name = "final_layer" if layer_idx == -1 else f"layer_{layer_idx}"
    output_path = os.path.join(
        output_dir, f"{model_short_name}_persona_feature_reps_{layer_name}.pt"
    )

    torch.save(representations, output_path)
    logger.info(
        f"Saved persona feature representations from {layer_name} to {output_path}"
    )


def process_persona_dataset(user, dataset_path, exp_config=None, data_fraction=1.0):
    """Process the persona dataset for a specific user level.

    Args:
        user (str): User level (expert, beginner, etc.)
        dataset_path (str): Path to save the processed dataset
        exp_config (ExperimentConfig): Experiment configuration object
        data_fraction (float): Fraction of data to keep (default: 1.0)
    """
    # Get random seed from config or use default
    random_seed = exp_config.random_seed if exp_config is not None else 42

    # Load dataset
    logger.info(f"Loading Persona dataset for user level: {user}")
    try:
        dataset = load_dataset(
            exp_config.dataset_name if exp_config is not None else "SynthLabsAI/PERSONA"
        )
        logger.info("Dataset loaded successfully!")

        # Log dataset structure for debugging
        logger.info(f"Dataset keys: {dataset.keys()}")
        logger.info(f"Train dataset columns: {dataset['train'].column_names}")
        logger.info(f"Train dataset size: {len(dataset['train'])}")

        # Log a few examples to understand the structure
        logger.info(f"Sample data: {dataset['train'][:2]}")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

    # Convert to DataFrame
    df = pd.DataFrame(dataset["train"])

    # Log column statistics
    logger.info(f"DataFrame columns: {df.columns.tolist()}")
    if "scorer_level" in df.columns:
        logger.info(f"Unique scorer_level values: {df['scorer_level'].unique()}")

    # Remove unwanted columns if any
    columns_to_keep = [
        "data",  # persona's response to question
        "persona",  # persona's description
        "instruction",  # question
        "original",  # original response (LLM without persona)
        "question type",  # train, test
        # not included: critique,
    ]
    df = df[columns_to_keep]

    # This dataset, although labeled as "train" in huggingface, is the full dataset
    # Filter only on train split
    df = df[df["question type"] == "train"]

    # Create the "instruction+data" column by concatenating instruction and data
    df["instruction+data"] = df["instruction"] + " " + df["data"]
    logger.info(
        "Created 'instruction+data' column by concatenating instruction and data"
    )

    # Create a mapping from unique persona strings to indices
    unique_personas = df["persona"].unique()
    persona_to_idx = {persona: idx for idx, persona in enumerate(unique_personas)}

    # Add persona_idx column to the DataFrame
    df["persona_idx"] = df["persona"].map(persona_to_idx)

    logger.info(f"Added persona_idx column with {len(persona_to_idx)} unique values")

    # Log a few examples of the mapping
    sample_personas = list(persona_to_idx.items())[:3]
    for persona, idx in sample_personas:
        logger.info(f"Persona (truncated) '{persona[:50]}...' -> idx {idx}")

    # Extract persona features and add them as columns
    df = extract_persona_features(df)

    # Sample a subset of data if data_fraction < 1.0
    if data_fraction < 1.0:
        logger.info(f"Sampling {data_fraction * 100}% of the data")
        df = df.sample(frac=data_fraction, random_state=random_seed)

    # Save
    save_processed_dataset(df, dataset_path)
    return df


def save_processed_dataset(df, path):
    """Save the processed dataset."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Shuffle and save
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(path, index=False)
    logger.info(f"Reorganized dataset saved to {path}")

    # Print statistics
    logger.info("Reorganized dataset columns:")
    logger.info(df.columns.tolist())

    logger.info("Number of unique personas:")
    logger.info(df["persona"].nunique())

    logger.info("Number of unique questions:")
    logger.info(df["instruction"].nunique())

    # Log information about the new column
    logger.info("Sample of instruction+data column:")
    logger.info(df["instruction+data"].head(2).tolist())
