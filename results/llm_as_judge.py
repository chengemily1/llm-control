import os
import json
import yaml
import openai
from typing import Dict, List, Tuple
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pathlib import Path
import argparse

# Load environment variables from .env file in parent directory
load_dotenv(Path(__file__).parent.parent / '.env')

# Load evaluation prompt
with open('results/eval_prompt.txt', 'r') as f:
    EVAL_PROMPT = f.read()

# Load configuration
with open('results/eval_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize OpenAI client
client = AsyncOpenAI(
    api_key=os.getenv('OPENAI_API_KEY')  # Make sure to set this environment variable
)

async def evaluate_pair(
    question: str,
    response_a: str,
    response_b: str,
    user_description: str
) -> str:
    """
    Evaluate a pair of responses using GPT-4 and return the better model identifier.
    Returns 'm', 'M', or None (for tie/unsure)
    """
    comparison_data = f"""Question: {question}
Response A: {response_a}
Response B: {response_b}
User: {user_description}

If you are unsure which response is better, you can output 'tie' to indicate no clear winner."""

    try:
        response = await client.chat.completions.create(
            model="gpt-4-turbo-preview",  # Using GPT-4 Turbo
            messages=[
                {"role": "system", "content": EVAL_PROMPT},
                {"role": "user", "content": comparison_data}
            ],
            temperature=0,  # We want deterministic responses
            max_tokens=10   # We only need a single character response
        )
        
        # Extract the model choice ('m', 'M', or 'tie')
        model_choice = response.choices[0].message.content.strip().lower()
        if model_choice == 'tie':
            return None
        elif model_choice in ['m', 'M']:
            return model_choice
        else:
            print(f"Unexpected response: {model_choice}, treating as tie")
            return None
    
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None

async def evaluate_results(baseline_path: str, controlled_path: str) -> Dict:
    """
    Evaluate pairs of responses from baseline and controlled experiments.
    """
    # Load results
    with open(baseline_path, 'r') as f:
        baseline_results = json.load(f)
    with open(controlled_path, 'r') as f:
        controlled_results = json.load(f)

    evaluation_results = {}
    
    # Process each pair of responses
    for question_id in baseline_results:
        if question_id in controlled_results:
            baseline_response = baseline_results[question_id]["response"]
            controlled_response = controlled_results[question_id]["response"]
            question = baseline_results[question_id]["question"]
            user_description = baseline_results[question_id]["user_description"]
            
            # Randomly assign responses to 'm' and 'M' to avoid bias
            import random
            if random.random() < 0.5:
                result = await evaluate_pair(question, baseline_response, controlled_response, user_description)
                mapping = {'m': 'baseline', 'M': 'controlled'}
            else:
                result = await evaluate_pair(question, controlled_response, baseline_response, user_description)
                mapping = {'m': 'controlled', 'M': 'baseline'}
            
            winner = 'tie' if result is None else mapping[result]
            evaluation_results[question_id] = {
                'winner': winner,
                'question': question,
                'baseline_response': baseline_response,
                'controlled_response': controlled_response,
                'user_description': user_description,
                'mapping': mapping  # Store which model was 'm' and which was 'M'
            }

    return evaluation_results

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate model outputs using GPT-4')
    parser.add_argument('--user', type=str, required=True,
                       choices=['child', 'preteen', 'teenager', 'young adult', 'expert'],
                       help='User type for evaluation')
    return parser.parse_args()

async def main():
    args = parse_arguments()
    
    # Paths to your result files
    baseline_path = 'results/llama-3-8b-baseline-results.json'
    controlled_path = 'results/llama-3-8b-controlled-results.json'
    
    # Run evaluation
    results = await evaluate_results(baseline_path, controlled_path)
    
    # Save evaluation results with user type in filename
    output_path = f'results/evaluation_results_{args.user}.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    total = len(results)
    baseline_wins = sum(1 for r in results.values() if r['winner'] == 'baseline')
    controlled_wins = sum(1 for r in results.values() if r['winner'] == 'controlled')
    ties = sum(1 for r in results.values() if r['winner'] == 'tie')
    
    print(f"\nEvaluation Summary for {args.user}:")
    print(f"Total comparisons: {total}")
    print(f"Baseline wins: {baseline_wins} ({baseline_wins/total*100:.1f}%)")
    print(f"Controlled wins: {controlled_wins} ({controlled_wins/total*100:.1f}%)")
    print(f"Ties: {ties} ({ties/total*100:.1f}%)")
    
    # Save detailed results about which model was which letter
    mapping_path = f'results/evaluation_mappings_{args.user}.json'
    mappings = {qid: result['mapping'] for qid, result in results.items()}
    with open(mapping_path, 'w') as f:
        json.dump(mappings, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main()) 