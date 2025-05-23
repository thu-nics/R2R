import pandas as pd
import sys
import argparse
from r2r.evaluate.eval_utils import get_answer_extractor, check_answer_correctness
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

def process_csv(input_file, output_file=None, answer_type="boxed"):
    """
    Process the CSV file and extract answers from model outputs.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str, optional): Path to output CSV file. If None, will append '_processed' to input filename
        answer_type (str): Type of answer to extract (default: "boxed")
    """
    # Initialize tokenizer
    print("Initializing DeepSeek tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", trust_remote_code=True)
    
    # Read the CSV file
    print("Reading input CSV file...")
    df = pd.read_csv(input_file)
    
    # Load AIME dataset from Hugging Face
    print("Loading AIME dataset...")
    if '24' in input_file:
        aime_dataset = load_dataset("Maxwell-Jia/AIME_2024")
    else:
        aime_dataset = load_dataset("di-zhang-fdu/AIME_1983_2024")
    # Convert to pandas DataFrame for easier matching
    aime_df = pd.DataFrame(aime_dataset['train'])
    
    # Get the appropriate answer extractor
    answer_extractor = get_answer_extractor(answer_type)
    
    # Process each row
    results = []
    print("Processing model outputs...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        model_output = row['model_output']
        
        # Count tokens in model output
        token_count = len(tokenizer.encode(model_output))
        
        # Extract answer from model output
        if '</think>' in model_output:
            model_answer = model_output.split('</think>')[1]
        else:
            model_answer = model_output
            
        # Extract predicted answer using the appropriate extractor
        predicted_answer, has_answer = answer_extractor(model_answer)
        
        # Find matching row in AIME dataset
        matching_row = aime_df[aime_df['ID'] == row['ID']]
        actual_answer = matching_row['Answer'].iloc[0] if not matching_row.empty else None
        
        # Check correctness if we have both predicted and actual answers
        is_correct = None
        if has_answer and actual_answer is not None:
            is_correct = check_answer_correctness(predicted_answer, actual_answer, answer_type)
        
        # Store results
        results.append({
            'index': idx,
            'ID': row['ID'],
            'model_output': model_output,
            'predicted_answer': predicted_answer,
            'has_answer': has_answer,
            'actual_answer': actual_answer,
            'is_correct': is_correct,
            'token_count': token_count
        })
    
    # Convert results to DataFrame
    print("Saving results...")
    results_df = pd.DataFrame(results)
    
    # Save to output file
    if output_file is None:
        output_file = input_file.replace('.csv', '_processed.csv')
    
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Print token statistics
    total_tokens = results_df['token_count'].sum()
    avg_tokens = results_df['token_count'].mean()
    print(f"\nToken Statistics:")
    print(f"Total tokens: {total_tokens}")
    print(f"Average tokens per output: {avg_tokens:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process CSV file and extract answers from model outputs')
    parser.add_argument('--input', required=True, help='Path to input CSV file')
    parser.add_argument('--output', help='Path to output CSV file (optional)')
    parser.add_argument('--answer_type', default='boxed', choices=['boxed', 'multiple_choice', 'code', 'mmlu-multiple-choice'],
                      help='Type of answer to extract (default: boxed)')
    
    args = parser.parse_args()
    process_csv(args.input, args.output, args.answer_type)
