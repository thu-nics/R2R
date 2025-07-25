"""
Step3 verifies continuation pairs using a specified verification model.
It takes a CSV file containing these pairs, processes them in batches, and appends the verification results to a new output CSV file.

Inputs:
- The CSV file (specified by `--input_csv`) generated by the previous step (i.e., `step_2_llm_continuation.py`). 
  This file must contain columns:
    - `small_diverge_text`: The continuation generated from the Small Language Model (SLM) token.
    - `reference_diverge_text`: The continuation generated from the Large Language Model (LLM) token.
    - `common_context`: The shared context provided to both models for generating the continuations.
    - Optionally, `data_id` and `token_id` can be present for tracking.
- Command-line arguments to control:
    - Verification model parameters (`--verify_model`, `--verify_mode`, `--tp_size`, `--mem_fraction`).
    - Processing parameters (`--batch_size`, `--save_interval`).

Outputs:
- An output CSV file.
- This CSV file includes all columns from the input file, plus two new columns:
    - `divergent`: A score (e.g., similarity score) from the verification model indicating the degree of divergence or a binary judgment.
    - `verify_response`: The raw textual response or justification from the verification model.
- If `--save_interval` is used, results are saved periodically; otherwise, they are saved at the end of processing.
"""

import argparse
import pandas as pd
from tqdm import tqdm
import torch
from typing import List
import math
import os
import json

from r2r.data.generation_controller import DivergePoint
from r2r.data.verify_model import VerifyModel
from r2r.utils.config import MODEL_DICT

def parse_args():
    parser = argparse.ArgumentParser(description='verify CSV with divergent text pairs')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output_csv', type=str, default=None, 
                        help='Path to output CSV file')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size for processing')
    parser.add_argument('--verify_model', type=str, default=None, help='Verify model to use. If not provided, will use the model specified in r2r/utils/config.py')
    parser.add_argument('--verify_mode', type=str, default='common_context',
                        choices=['common_context'],
                        help='Judgment mode to use for evaluation')
    parser.add_argument('--tp_size', type=int, default=4,
                        help='Tensor parallel size')
    parser.add_argument('--mem_fraction', type=float, default=0.9,
                        help='Memory fraction for verify model')
    parser.add_argument('--save_interval', type=int, default=100,
                        help='Save results every N batches. 0 means save only at the end.')
    return parser.parse_args()

def convert_row_to_diverge_point(row):
    """Convert a DataFrame row to a DivergePoint object."""
    return DivergePoint(
        data_id=row.get('data_id', 0),
        token_id=row.get('token_id', 0),
        small_diverge_text=row['small_diverge_text'],
        reference_diverge_text=row['reference_diverge_text'],
        common_context=row['common_context'],
        pred_small_token=row.get('pred_small_token', []),
        pred_small_text=row.get('pred_small_text', ''),
    )

def save_results_to_csv(df_to_save, output_csv, mode='w', header=True):
    """Saves a DataFrame to a CSV file with specified mode and header settings."""
    print(f"Saving results to {output_csv} (mode: {mode}, header: {header})...")
    try:
        df_to_save.to_csv(output_csv, mode=mode, header=header, index=False)
        print(f"Results successfully saved to {output_csv}")
    except Exception as e:
        print(f"Error saving results to {output_csv}: {e}")

def handle_periodic_save(args, results_to_save, batch_idx, num_batches, is_first_save):
    """Handles periodic saving of results during processing."""
    if args.save_interval <= 0 or not results_to_save:
        return results_to_save, is_first_save # No periodic saving or nothing to save

    is_last_batch = (batch_idx == num_batches - 1)
    should_save_now = ((batch_idx + 1) % args.save_interval == 0) or is_last_batch

    if should_save_now:
        print(f"\nProcessing results after batch {batch_idx + 1} for periodic saving...")
        save_df = pd.DataFrame(results_to_save)
        write_header = not os.path.exists(args.output_csv) or is_first_save
        save_results_to_csv(save_df, args.output_csv, mode='a', header=write_header)
        results_to_save = [] # Clear buffer
        is_first_save = False # Header has been handled for subsequent appends
    
    return results_to_save, is_first_save

def handle_final_save(args, df, all_scores, verify_responses):
    """Handles the final save operation if periodic saving was not used."""
    if args.save_interval == 0:
        print("Processing final results...")
        # Ensure columns exist before assigning
        if 'divergent' not in df.columns:
            df['divergent'] = None
        if 'verify_response' not in df.columns:
            df['verify_response'] = None
            
        # Check length compatibility before assignment
        if len(all_scores) == len(df) and len(verify_responses) == len(df):
            df['divergent'] = all_scores
            df['verify_response'] = verify_responses
            save_results_to_csv(df, args.output_csv, mode='w', header=True) # Overwrite
        else:
            print(f"Error: Length mismatch. Scores ({len(all_scores)}), Responses ({len(verify_responses)}), DataFrame ({len(df)}). Cannot perform final save.")

def main():
    args = parse_args()

    # Decide output csv
    if args.output_csv is None:
        args.output_csv = args.input_csv.replace('.csv', '_verify.csv')
    
    # Load the CSV file
    print(f"Loading CSV from {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    
    # Verify required columns exist
    required_columns = ['small_diverge_text', 'reference_diverge_text', 'common_context']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {missing_columns}")
    
    # Initialize the verify model
    verify_model = VerifyModel(
        model_name=MODEL_DICT["verify"]["model_path"],
        verify_mode=args.verify_mode,
        max_new_tokens=MODEL_DICT["verify"]["max_new_tokens"],
        mem_fraction_static=args.mem_fraction,
        tp_size=args.tp_size,
        apply_chat_template_kwargs=getattr(MODEL_DICT["verify"], "apply_chat_template_kwargs", None)
    )
    
    # Process the data in batches
    total_rows = len(df)
    num_batches = math.ceil(total_rows / args.batch_size)
    
    # Prepare for results
    all_scores = []
    verify_responses = []
    results_to_save = [] # For periodic saving
    is_first_save = True # For header management when appending
    
    print(f"Processing {total_rows} rows in {num_batches} batches")
    
    for batch_idx in tqdm(range(num_batches), desc="Judging batches"):
        # Get current batch
        start_idx = batch_idx * args.batch_size
        end_idx = min((batch_idx + 1) * args.batch_size, total_rows)
        batch_df = df.iloc[start_idx:end_idx]
        
        # Convert batch rows to DivergePoint objects
        batch_diverge_points = [convert_row_to_diverge_point(row) for _, row in batch_df.iterrows()]
        
        # Process the batch
        batch_comparison_points = verify_model.batch_compare_diverge_points(batch_diverge_points)
        
        # Extract results
        batch_scores = [point.similarity_score for point in batch_comparison_points]
        batch_responses = [point.verify_response for point in batch_comparison_points]
        
        # Store results for final dataframe if not saving periodically
        if args.save_interval == 0:
            all_scores.extend(batch_scores)
            verify_responses.extend(batch_responses)
        else:
            # Prepare results for periodic saving
            for i, (_, original_row) in enumerate(batch_df.iterrows()):
                row_dict = original_row.to_dict()
                row_dict['divergent'] = batch_scores[i]
                row_dict['verify_response'] = batch_responses[i]
                results_to_save.append(row_dict)

        # Handle periodic saving
        results_to_save, is_first_save = handle_periodic_save(args, results_to_save, batch_idx, num_batches, is_first_save)
    
    # Handle final save if necessary (i.e., if save_interval == 0)
    handle_final_save(args, df, all_scores, verify_responses)
    
    # Clean up
    verify_model.shutdown()
    print("Done!")

if __name__ == "__main__":
    main()