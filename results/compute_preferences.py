import json
import os

def compute_preference_percentage(eval_results_path: str) -> float:
    """Compute percentage of times controlled was preferred over baseline."""
    with open(eval_results_path, 'r') as f:
        results = json.load(f)
    
    total = len(results)
    if total == 0:
        return 0.0
        
    controlled_wins = sum(1 for r in results.values() if r['winner'] == 'controlled')
    return (controlled_wins / total) * 100

def main():
    # List of user types
    user_types = ['child', 'preteen', 'teenager', 'young adult', 'expert']
    
    # Dictionary to store results
    preference_results = {}
    
    # Process results for each user type
    for user in user_types:
        eval_path = f'results/evaluation_results_{user}.json'
        if os.path.exists(eval_path):
            percentage = compute_preference_percentage(eval_path)
            preference_results[user] = percentage
    
    # Create experiments/elix directory if it doesn't exist
    os.makedirs('experiments/elix', exist_ok=True)
    
    # Save results in experiments/elix directory
    with open('experiments/elix/RESULTS.json', 'w') as f:
        json.dump(preference_results, f, indent=2)
    
    # Print results
    print("\nPreference Results:")
    print("------------------")
    for user, percentage in preference_results.items():
        print(f"{user}: {percentage:.1f}%")

if __name__ == "__main__":
    main() 