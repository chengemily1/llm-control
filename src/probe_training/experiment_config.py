"""Configuration management for probe training experiments."""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class BaseExperimentConfig:
    """Base configuration for probe training experiments."""
    # Model settings
    model_name: str = "meta-llama/Meta-Llama-3-8B"
    device: str = "cuda"
    
    # Training settings
    batch_size: int = 1
    num_epochs: int = 1000
    learning_rate: float = 0.001
    random_seed: int = 42
    downsample: float = 1.0
    
    def get_paths(self, base_path: str) -> dict:
        """Get all relevant paths for the experiment."""
        exp_dir = os.path.join(base_path, "experiments", self.get_experiment_name())
        
        return {
            "data": {
                "base": os.path.join(exp_dir, "data"),
                "processed": self.get_processed_data_path(exp_dir),
            },
            "representations": {
                "base": os.path.join(exp_dir, "saved_reps"),
                "layer": lambda layer: os.path.join(
                    exp_dir, "saved_reps",
                    f"{self.model_name.split('/')[-1]}_reps_part_final.pt"
                ),
            },
            "probes": {
                "base": os.path.join(exp_dir, "saved_probes"),
                "results": os.path.join(exp_dir, "probing_results"),
            }
        }
    
    def get_probe_name(self, layer: int) -> str:
        """Get the name for a probe at a specific layer."""
        model_name = self.model_name.split('/')[-1]
        ds_str = f"_downsample_{self.downsample}" if self.downsample < 1 else ""
        return f"{model_name}_linear_probe_layer_{layer}_rs{self.random_seed}{ds_str}"
    
    def get_results_name(self, layer: int) -> str:
        """Get the name for results at a specific layer."""
        model_name = self.model_name.split('/')[-1]
        ds_str = f"_downsample_{self.downsample}" if self.downsample < 1 else ""
        return f"{model_name}_layer_{layer}_rs{self.random_seed}{ds_str}_validation_results_over_training.json"
    
    def get_experiment_name(self) -> str:
        """Get the name of the experiment. Must be implemented by child classes."""
        raise NotImplementedError("Child classes must implement get_experiment_name")
    
    def get_processed_data_path(self, exp_dir: str) -> str:
        """Get the path to the processed data file. Must be implemented by child classes."""
        raise NotImplementedError("Child classes must implement get_processed_data_path")
    
    def validate_config(self):
        """Validate configuration. Can be overridden by child classes."""
        pass

@dataclass
class ElixExperimentConfig(BaseExperimentConfig):
    """Configuration for Elix experiment."""
    # Data settings
    text_field: str = "prompt_response"  # Field to use from dataset
    label_field: str = "score"  # Field to use as labels
    
    # User settings
    user: str = "preteen"  # User type for elix experiment
    
    def __post_init__(self):
        """Validate and set derived attributes."""
        self.validate_config()
        
        # Set text field based on experiment
        if self.get_experiment_name() != "elix":
            self.text_field = "prompt"
    
    def validate_config(self):
        """Validate Elix-specific configuration."""
        valid_users = ["child", "preteen", "teenager", "young adult", "expert"]
        if self.user not in valid_users:
            raise ValueError(f"Invalid user type: {self.user}. Must be one of {valid_users}")
    
    def get_experiment_name(self) -> str:
        """Get the name of the experiment."""
        return "elix"
    
    def get_processed_data_path(self, exp_dir: str) -> str:
        """Get the path to the processed data file."""
        return os.path.join(exp_dir, self.user, "data", "train_shuffled_balanced.csv")

@dataclass
class ReviewsExperimentConfig(BaseExperimentConfig):
    """Configuration for steered reviews experiment."""
    # Data settings
    text_field: str = "prompt_response"  # Field containing the review text
    label_field: str = "score"  # Field containing the preference score
    
    # Dataset settings
    dataset_name: str = "Asap7772/steered_reviews_full_autolabel_gpt4o_pref"
    split: str = "train"  # Dataset split to use
    user: str = "negative"  # User ID
    reps_suffix: str = "final"  # Which representation file to use (final or 5000)
    
    def __post_init__(self):
        """Validate and set derived attributes."""
        self.validate_config()
    
    def validate_config(self):
        """Validate reviews-specific configuration."""
        valid_splits = ["train", "validation", "test"]
        if self.split not in valid_splits:
            raise ValueError(f"Invalid split: {self.split}. Must be one of {valid_splits}")
        
        valid_reps = ["final", "5000"]
        if self.reps_suffix not in valid_reps:
            raise ValueError(f"Invalid reps_suffix: {self.reps_suffix}. Must be one of {valid_reps}")
    
    def get_experiment_name(self) -> str:
        """Get the name of the experiment."""
        return "reviews"
    
    def get_processed_data_path(self, exp_dir: str) -> str:
        """Get the path to the processed data file."""
        return os.path.join(exp_dir, f"user_{self.user}", "data", f"{self.split}_processed.csv")
    
    def get_paths(self, base_path: str) -> dict:
        """Get all relevant paths for the experiment."""
        exp_dir = os.path.join(base_path, "experiments", self.get_experiment_name())
        user_dir = os.path.join(exp_dir, f"user_{self.user}")
        
        return {
            "data": {
                "base": os.path.join(exp_dir, "data"),
                "processed": self.get_processed_data_path(exp_dir),
            },
            "representations": {
                "base": os.path.join(user_dir, "saved_reps"),
                "layer": lambda layer: os.path.join(
                    user_dir, "saved_reps",
                    f"{self.model_name.split('/')[-1]}_reps_part_{self.reps_suffix}.pt"
                ),
            },
            "probes": {
                "base": os.path.join(user_dir, "saved_probes"),
                "results": os.path.join(user_dir, "probing_results"),
            }
        }

# For backward compatibility
ExperimentConfig = ElixExperimentConfig 