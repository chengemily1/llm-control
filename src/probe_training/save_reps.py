"""Script to save model representations for probe training."""

import os
import sys
import argparse
import torch
from tqdm import tqdm
import pandas as pd

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.model_utils import setup_model_and_tokenizer
from src.utils.data_utils import encode_data
from src.utils.config_utils import load_config
from src.utils.dataset_utils_elix import process_elix_dataset
from src.utils.dataset_utils_reviews import process_reviews_dataset
from src.utils.dataset_utils_persona import process_persona_dataset
from src.utils.representation_utils import (
    save_layer_representations,
    save_attention_representations,
)
from src.probe_training.experiment_config import (
    ElixExperimentConfig,
    ReviewsExperimentConfig,
    PersonaExperimentConfig,
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Save model representations for probe training"
    )
    # Optional arguments that override experiment config
    parser.add_argument("--model_name", type=str, help="Override default model name")
    parser.add_argument(
        "--experiment",
        type=str,
        default="reviews",
        help="Experiment type (elix, reviews, or persona)",
    )
    parser.add_argument("--device", type=str, help="Override default device")
    parser.add_argument("--batch_size", type=int, help="Override default batch size")
    parser.add_argument("--random_seed", type=int, help="Override default random seed")

    # Other arguments
    parser.add_argument(
        "--attn", type=int, default=0, help="Whether to save attention representations"
    )
    parser.add_argument(
        "--data_fraction",
        type=float,
        help="Fraction of data to use",
        default=1.0,
    )
    parser.add_argument("--user")
    parser.add_argument(
        "--config", default="src/config.json", help="Path to config file"
    )
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_arguments()

    # Create experiment config based on experiment type
    if args.experiment == "reviews":
        exp_config = ReviewsExperimentConfig()
    elif args.experiment == "persona":
        exp_config = PersonaExperimentConfig()
    elif args.experiment == "elix":
        exp_config = ElixExperimentConfig()
    else:
        raise ValueError(f"Invalid experiment type: {args.experiment}")

    # Update config with any provided args
    for key, value in vars(args).items():
        if value is not None and hasattr(exp_config, key):
            setattr(exp_config, key, value)

    # Load configuration
    config = load_config(args.config)

    # Get paths
    paths = exp_config.get_paths(config["base_path"])

    # Process dataset if needed
    if not os.path.exists(paths["data"]["processed"]):
        if exp_config.get_experiment_name() == "elix":
            process_elix_dataset(
                exp_config.user,
                paths["data"]["processed"],
                exp_config,
                data_fraction=args.data_fraction,
            )
        elif exp_config.get_experiment_name() == "reviews":
            process_reviews_dataset(
                exp_config.user,
                paths["data"]["processed"],
                exp_config,
                data_fraction=args.data_fraction,
            )
        elif exp_config.get_experiment_name() == "persona":
            process_persona_dataset(
                exp_config.user,
                paths["data"]["processed"],
                exp_config,
                data_fraction=args.data_fraction,
            )
        else:
            raise ValueError(f"Invalid experiment type: {args.experiment}")

    # Load processed dataset
    dataset = pd.read_csv(paths["data"]["processed"])
    data = list(dataset[exp_config.text_field])

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        exp_config.model_name, config["access_token"], exp_config.device
    )

    # JL 3/19/25: deleted attention flag

    # Encode data in batches
    print("Encoding data")
    encoded_data = [
        encode_data(
            tokenizer,
            1,
            data[i : i + exp_config.batch_size],
            1,
            model.config.max_position_embeddings,
            exp_config.device,
        )[0]
        for i in tqdm(range(0, len(data), exp_config.batch_size))
    ]
    print("Encoding complete")

    # Save layer representations
    os.makedirs(paths["representations"]["base"], exist_ok=True)
    save_layer_representations(
        model,
        tokenizer,
        encoded_data,
        exp_config.batch_size,
        exp_config.device,
        paths["representations"]["base"],
        exp_config.model_name,
    )


if __name__ == "__main__":
    # Use: python src/probe_training/save_reps.py --experiment reviews --user 1
    # Or: python src/probe_training/save_reps.py --experiment persona
    main()
