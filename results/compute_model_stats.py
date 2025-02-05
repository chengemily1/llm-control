import pandas as pd
import numpy as np
import sys

# Check if the input file is provided as a command-line argument
if len(sys.argv) != 2:
    print("Usage: python compute_model_stats.py <input_file.csv>")
    sys.exit(1)

# Get the input file name from the command-line argument
input_file = sys.argv[1]

# Function to compute mean and std for a list of columns
def compute_stats(df, columns):
    stats = {}
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        stats[col] = {'mean': mean, 'std': std}
    return stats

# Load the CSV file
df = pd.read_csv(input_file)

# Dynamically determine the base_models by finding columns with both _human and _score
all_columns = df.columns
base_models = set()

for col in all_columns:
    if col.endswith('_human'):
        base_model = col[:-6]  # Remove the '_human' suffix
        score_col = f"{base_model}_score"
        if score_col in all_columns:
            base_models.add(base_model)

# Convert the set to a sorted list for consistency
base_models = sorted(base_models)

# Generate score and human columns by appending suffixes
score_columns = [f"{model}_score" for model in base_models]
human_columns = [f"{model}_human" for model in base_models]

# Compute stats for score and human columns
score_stats = compute_stats(df, score_columns)
human_stats = compute_stats(df, human_columns)

# Prepare the output DataFrame
output_data = []

# Iterate through each model and collect stats
for model in base_models:
    # Construct column names for scores and human evaluations
    score_key = f"{model}_score"
    human_key = f"{model}_human"
    
    # Get stats for scores and human evaluations
    score_mean = score_stats.get(score_key, {}).get('mean', np.nan)
    score_std = score_stats.get(score_key, {}).get('std', np.nan)
    human_mean = human_stats.get(human_key, {}).get('mean', np.nan)
    human_std = human_stats.get(human_key, {}).get('std', np.nan)
    
    # Append the results to the output data
    output_data.append({
        'model': model,
        'score_mean': score_mean,
        'score_std': score_std,
        'human_mean': human_mean,
        'human_std': human_std
    })

# Create a DataFrame from the output data
output_df = pd.DataFrame(output_data)

# Return the DataFrame
output_df

# Print the output DataFrame as a CSV string
print(output_df.to_csv(index=False))
