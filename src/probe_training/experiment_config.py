"""Configuration class for probe training experiments."""

from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class ExperimentConfig:
    """Configuration for probe training experiments."""
    
    # Model configuration
    model_name: str = "meta-llama/Llama-2-7b-hf"
    device: str = "cuda"
    batch_size: int = 32
    
    # Training configuration
    num_epochs: int = 1000
    learning_rate: float = 1e-3
    
    # Experiment configuration
    experiment: str = "gms8k"
    random_seed: int = 42
    rank: int = 8  # Number of dimensions for the probe output
    
    # Text field configuration
    text_field: str = "question_answer"
    
    def get_paths(self, base_path: str) -> Dict[str, Dict[str, str]]:
        """Get all necessary paths for the experiment.
        
        Args:
            base_path (str): Base path for the project
            
        Returns:
            Dict[str, Dict[str, str]]: Dictionary containing all necessary paths
        """
        return {
            'data': {
                'raw': f"{base_path}/experiments/{self.experiment}/raw",
                'processed': f"{base_path}/experiments/{self.experiment}/processed",
            },
            'representations': {
                'base': f"{base_path}/experiments/{self.experiment}/representations",
                'layers': f"{base_path}/experiments/{self.experiment}/representations/layers",
                'attention': f"{base_path}/experiments/{self.experiment}/representations/attention",
            },
            'probes': {
                'base': f"{base_path}/experiments/{self.experiment}/saved_probes",
            }
        } 