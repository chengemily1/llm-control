import os
import json

def load_config(config_path='src/config.json'):
    """Load configuration from JSON file or environment variables."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Get values from config or fall back to environment variables
        access_token = config.get('hf_access_token', os.environ.get('HF_ACCESS_TOKEN'))
        base_path = config.get('base_path', os.getcwd())
        
        if not access_token:
            raise ValueError("Please set HF_ACCESS_TOKEN in config.json or as environment variable")
            
        return {
            'access_token': access_token,  # Return as access_token for consistency
            'base_path': base_path
        }
        
    except FileNotFoundError:
        print(f"Config file {config_path} not found, using environment variables...")
        access_token = os.environ.get('HF_ACCESS_TOKEN')
        if not access_token:
            raise ValueError("Please set HF_ACCESS_TOKEN environment variable or provide it in config.json")
            
        return {
            'access_token': access_token,
            'base_path': os.getcwd()
        } 