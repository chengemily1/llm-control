"""Configuration management for probe training experiments."""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ExperimentConfig:
    """Configuration for probe training experiments."""
    # Model settings
    model_name: str = "meta-llama/Meta-Llama-3-8B"
    experiment: str = "elix"
    device: str = "cuda"
    
    # Training settings
    batch_size: int = 1
    num_epochs: int = 1000
    learning_rate: float = 0.0001
    random_seed: int = 42
    downsample: float = 1.0
    
    # Data settings
    text_field: str = "prompt_response"  # Field to use from dataset
    label_field: str = "score"  # Field to use as labels
    
    # User settings
    user: str = "child"  # User type for elix experiment
    
    def __post_init__(self):
        """Validate and set derived attributes."""
        if self.experiment not in ["elix"]:
            raise ValueError(f"Unknown experiment type: {self.experiment}")
        
        # Validate user type
        valid_users = ["child", "preteen", "teenager", "young adult", "expert"]
        if self.user not in valid_users:
            raise ValueError(f"Invalid user type: {self.user}. Must be one of {valid_users}")
        
        # Set text field based on experiment
        if self.experiment != "elix":
            self.text_field = "prompt"
    
    def get_paths(self, base_path: str) -> dict:
        """Get all relevant paths for the experiment."""
        exp_dir = os.path.join(base_path, "experiments", self.experiment, self.user)
        
        return {
            "data": {
                "base": os.path.join(exp_dir, "data"),
                "processed": os.path.join(exp_dir, "data", "train_shuffled_balanced.csv"),
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