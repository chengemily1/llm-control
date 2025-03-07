import json
import pandas as pd
import argparse
import os

def load_results(file_path):
    """Load results from JSON file and convert to DataFrame."""
    with open(file_path, 'r') as f:
        results = json.load(f)
    
    # Extract questions and generated texts
    data = []
    for question, content in results.items():
        # Skip non-dictionary entries (like 'overall_latency')
        if not isinstance(content, dict) or 'generated_text' not in content:
            continue
            
        # Remove "Question: " prefix if it exists
        clean_question = question.replace("Question: ", "").split("\nAnswer:")[0]
        
        # Clean up generated text by removing Question/Answer prefix
        generated_text = content['generated_text']
        if "Question:" in generated_text and "\nAnswer:" in generated_text:
            generated_text = generated_text.split("\nAnswer:", 1)[1].strip()
        
        data.append({
            'question': clean_question,
            'generated_text': generated_text
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze generation results')
    parser.add_argument('--results_file', type=str, help='Path to results JSON file', default='/scr/biggest/carmen/llm-control/experiments/elix/control_results/Meta-Llama-3-8B_ours.json')
    parser.add_argument('--output_csv', type=str, help='Path to save CSV output (optional)', default='/scr/biggest/carmen/llm-control/experiments/elix/control_results/Meta-Llama-3-8B_ours.csv')
    args = parser.parse_args()

    # Load and convert results
    df = load_results(args.results_file)
    
    # Print basic statistics
    print(f"\nTotal number of questions: {len(df)}")
    print("\nFirst few entries:")
    print(df.head())
    
    breakpoint()
    # Save to CSV if requested
    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"\nResults saved to: {args.output_csv}") 