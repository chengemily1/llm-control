"""Utilities for processing the persona dataset."""

import pandas as pd
from datasets import load_dataset
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
