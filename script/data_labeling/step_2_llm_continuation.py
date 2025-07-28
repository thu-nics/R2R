"""
Step2 generates the LLM continuation based on the response prefix and the non-identical SLM-token predictions.

Inputs:
- Prediction comparison CSV file (specified by `--input_path`) from step 1. This file contains mismatch points between SLM and LLM tokens.

Outputs:
- A directory (specified by `--output_path`, default: `output/playground/continuation`) containing:
    - `args.json`: A JSON file logging the arguments used for the script execution.
    - `generation_results_data_all_real.csv`: Each row corresponds to a processed mismatch, detailing the generated continuation from the SLM token and the LLM continuation, along with their respective contexts.
"""

import os
from pathlib import Path
import pandas as pd
import logging
from tqdm import tqdm
import argparse
import sglang as sgl
from typing import Dict, Tuple, List
import math
import json
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from r2r.data.data_process import DataProcessor, MismatchPoint
from r2r.data.generation_controller import ModelController, DivergePoint
from r2r.data.verify_model import ComparisonPoint, data_points_to_df
from r2r.utils.config import TOKEN_TYPE, MODEL_DICT

def check_model_dict_consistency(input_path):
    """
    Check if MODEL_DICT is consistent with the models used in the input data
    
    Args:
        input_path: Path to the input CSV file
        
    Returns:
        bool: True if MODEL_DICT is consistent, False otherwise
        
    Raises:
        FileNotFoundError: If required args.json or run_args.json files are not found
        KeyError: If required keys are missing from the JSON files
    """
    input_path = Path(input_path)
    
    try:
        # Read args.json from parent directory (contains test_model_list)
        args_json_path = input_path.parent / 'args.json'
        if not args_json_path.exists():
            raise FileNotFoundError(f"args.json not found at {args_json_path}")
            
        with open(args_json_path, 'r') as f:
            args_data = json.load(f)
            
        if 'test_model_list' not in args_data or len(args_data['test_model_list']) == 0:
            raise KeyError("test_model_list not found or empty in args.json")
            
        slm_path_from_file = args_data['test_model_list'][0]
        
        # Read run_args.json from grandparent directory (contains model_path)
        run_args_json_path = input_path.parent.parent / 'run_args.json'
        if not run_args_json_path.exists():
            raise FileNotFoundError(f"run_args.json not found at {run_args_json_path}")
            
        with open(run_args_json_path, 'r') as f:
            run_args_data = json.load(f)
            
        if 'model_path' not in run_args_data:
            raise KeyError("model_path not found in run_args.json")
            
        llm_path_from_file = run_args_data['model_path']
        
        # Get paths from MODEL_DICT
        dict_slm_path = MODEL_DICT['quick']['model_path']
        dict_llm_path = MODEL_DICT['reference']['model_path']
        
        # Compare paths (normalize to handle different mount points)
        def normalize_path(path):
            """Normalize path by replacing different mount prefixes"""
            path = str(path)
            # Replace /mnt/public with /share/public for comparison
            if path.startswith('/mnt/public'):
                path = path.replace('/mnt/public', '/share/public')
            return Path(path)
        
        slm_match = normalize_path(slm_path_from_file) == normalize_path(dict_slm_path)
        llm_match = normalize_path(llm_path_from_file) == normalize_path(dict_llm_path)
        
        logger.info(f"Model consistency check:")
        logger.info(f"  SLM - File: {slm_path_from_file}")
        logger.info(f"  SLM - Dict: {dict_slm_path}")
        logger.info(f"  SLM - Match: {slm_match}")
        logger.info(f"  LLM - File: {llm_path_from_file}")
        logger.info(f"  LLM - Dict: {dict_llm_path}")
        logger.info(f"  LLM - Match: {llm_match}")
        
        if not slm_match:
            logger.warning(f"SLM model path mismatch! File: {slm_path_from_file}, MODEL_DICT: {dict_slm_path}")
        if not llm_match:
            logger.warning(f"LLM model path mismatch! File: {llm_path_from_file}, MODEL_DICT: {dict_llm_path}")
            
        return slm_match and llm_match
        
    except Exception as e:
        logger.error(f"Error checking MODEL_DICT consistency: {e}")
        return False

def parse_args():
    parser = argparse.ArgumentParser(description='Process data with specified data sample ID range')
    parser.add_argument('--input_path', type=str, default="output/playground/prediction_comparison.csv", help='Path to input CSV file. If not provided, will use default path.')
    parser.add_argument('--output_path', type=str, default='output/playground/continuation', help='Path to store output CSV files')

    parser.add_argument('--tp_size', type=int, default=2, help='Number of tensor parallel GPUs')
    parser.add_argument('--dp_size', type=int, default=1, help='Number of data parallel GPUs')
    parser.add_argument('--low', type=int, default=None, help='Lower bound of data sample ID range (inclusive). If not specified, will process all data.')
    parser.add_argument('--high', type=int, default=None, help='Upper bound of data sample ID range (exclusive). If not specified, will process all data.')
    parser.add_argument('--max_tokens', type=int, default=32768, help='Maximum number of tokens per data sample')
    parser.add_argument('--max_new_tokens', type=int, default=8192, help='Maximum number of new tokens to generate')
    parser.add_argument('--gen_mem_fraction', type=float, default=0.9,
                        help='Fraction of GPU memory to allocate for generation')
    parser.add_argument('--verify_mem_fraction', type=float, default=0.3,
                        help='Fraction of GPU memory to allocate for judging')
    parser.add_argument('--num_continuation', type=int, default=1, 
                        help='Number of continuations to generate (number of times to encounter EOS tokens)')
    parser.add_argument('--previous_context', type=int, default=0,
                        help='Number of previous sentences to include in the context')
    parser.add_argument('--common_previous_context', type=int, default=-1,
                        help='Number of previous sentences to include in the common context')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of small model outputs to generate for each mismatch point')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Temperature for sampling. Higher values produce more random samples.')
    parser.add_argument('--top_p', type=float, default=1.0,
                        help='Top-p probability threshold for nucleus sampling (0 < top_p ≤ 1)')
    parser.add_argument('--top_k', type=int, default=-1,
                        help='Top-k sampling parameter for generation')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of mismatches to batch together for generation')
    parser.add_argument('--skip_stress_divergent_token', action='store_true', default=False,
                        help='Whether to skip stressing the divergent token')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Whether to resume from existing results by skipping already processed data IDs')
    parser.add_argument('--is_print', default=False,
                        help='Whether to print the results')
    return parser.parse_args()

def get_processed_data_ids_and_full_results(results_path):
    """Read existing results file and return both processed data_ids and full results DataFrame"""
    if not results_path.exists():
        return set(), pd.DataFrame()
    
    try:
        existing_df = pd.read_csv(results_path)
        if 'data_id' in existing_df.columns:
            processed_ids = set(existing_df['data_id'].unique())
            logger.info(f"Found {len(processed_ids)} already processed data IDs in {results_path}")
            logger.info(f"Loaded {len(existing_df)} existing result rows")
            return processed_ids, existing_df
        else:
            logger.warning(f"No 'data_id' column found in existing results file {results_path}")
            return set(), pd.DataFrame()
    except Exception as e:
        logger.warning(f"Could not read existing results file {results_path}: {e}")
        return set(), pd.DataFrame()

def main():
    args = parse_args()
    
    # Check MODEL_DICT consistency with the models used in the input data
    logger.info("Checking MODEL_DICT consistency...")
    is_consistent = check_model_dict_consistency(args.input_path)
    if not is_consistent:
        logger.error("MODEL_DICT is not consistent with the models used in the input data!")
        logger.error("Please update MODEL_DICT configuration or use the correct input data.")
        raise ValueError("MODEL_DICT configuration mismatch detected")
    logger.info("MODEL_DICT consistency check passed.")
    
    # Use input_path from args if provided, otherwise use default
    input_path = Path(args.input_path)

    # Use output_path from args
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    # convert args to json and append MODEL_DICT
    args_json = args.__dict__.copy()
    args_json['MODEL_DICT'] = MODEL_DICT
    with open(output_path / 'args.json', 'w') as f:
        json.dump(args_json, f, indent=2)
    # Define output path at the start
    if args.low is not None and args.high is not None:
        results_filename = f"generation_results_data_{args.low}_to_{args.high}_real.csv"
    else:
        results_filename = f"generation_results_data_all_real.csv"

    results_path = output_path / results_filename
    logger.info(f"Results will be saved to {results_path}")
    logger.info(f"Using real tokens (ground truth) as comparison model")
    logger.info(f"Using batch size of {args.batch_size} for generation")
    
    stressing_format = lambda text: "<<" + text + ">>"

    # Load data and filter for data_id within specified range
    sample_data = pd.read_csv(input_path)

    # Filter data if low and high are specified
    if args.low is not None and args.high is not None:
        sample_data = sample_data[
            (sample_data['data_id'] >= args.low) & 
            (sample_data['data_id'] < args.high)
        ]
        print(f"Loaded {len(sample_data)} rows with data sample IDs in range [{args.low}, {args.high})")
    else:
        print(f"Loaded {len(sample_data)} rows with all data sample IDs")

    # Handle resume functionality
    existing_results_df = pd.DataFrame()
    if args.resume:
        logger.info("Resume mode enabled - checking for existing results...")
        processed_data_ids, existing_results_df = get_processed_data_ids_and_full_results(results_path)
        
        if processed_data_ids:
            # Filter out already processed data IDs
            original_count = len(sample_data)
            sample_data = sample_data[~sample_data['data_id'].isin(processed_data_ids)]
            remaining_count = len(sample_data)
            
            logger.info(f"Resume: Filtered out {original_count - remaining_count} already processed data IDs")
            logger.info(f"Resume: {remaining_count} data samples remaining to process")
            logger.info(f"Resume: Loaded {len(existing_results_df)} existing result rows that will be preserved")
            
            if remaining_count == 0:
                logger.info("All data samples have already been processed.")
                logger.info("Resume: Existing results are already complete, no new processing needed.")
                # Still save the final results file to ensure consistency
                if not existing_results_df.empty:
                    if args.low is not None and args.high is not None:
                        final_results_filename = f"generation_results_data_{args.low}_to_{args.high}_real_full.csv"
                    else:
                        final_results_filename = f"generation_results_data_all_real_full.csv"
                    final_results_path = output_path / final_results_filename
                    logger.info(f"Saving existing results to final file: {final_results_path}")
                    existing_results_df.to_csv(final_results_path, index=False)
                return
        else:
            logger.info("Resume: No existing results found, processing all data samples")
    else:
        logger.info("Resume mode disabled - processing all data samples (existing results may be overwritten)")
        existing_results_df = pd.DataFrame()

    # Process the data
    processor = DataProcessor(sample_data, max_tokens=args.max_tokens, comparison_model='real')

    # Initialize the generation controller and verify model
    gen_controller = ModelController(comparison_model='real', mem_fraction_static=args.gen_mem_fraction, tp_size=args.tp_size, dp_size=args.dp_size)
    
    logger.info("verify model disabled (--verify flag not provided)")
        
    # Test group_mismatches_by_data_id
    print("\nTesting group_mismatches_by_data_id:")
    grouped_mismatches = processor.group_mismatches_by_data_id()

    # Dictionary to store mismatch points by (data_id, token_id) for later reference
    mismatch_points_by_id = {}

    # Process all sentences and collect comparison points
    comparison_points = []
    
    # Track if we need to write existing results first (for resume mode)
    first_write_done = False
    
    for data_id, mismatches in tqdm(grouped_mismatches.items(), desc="Processing data samples"):
        data_comparison_points = []
        data_mismatch_points = {}
        logger.info(f"\nProcessing data sample {data_id} with {len(mismatches)} mismatches")

        # Get the data context for this data_id to access full real tokens
        data_context = processor.get_data_context(data_id)

        # Prepare lists to store generation results
        generation_results = []

        # Prepare batches of mismatches
        num_batches = math.ceil(len(mismatches) / args.batch_size)
        for batch_idx in range(num_batches):
            # Get current batch of mismatches
            start_idx = batch_idx * args.batch_size
            end_idx = min((batch_idx + 1) * args.batch_size, len(mismatches))
            batch_mismatches = mismatches[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_idx+1}/{num_batches} with {len(batch_mismatches)} mismatches")
            
            # Collect context tokens and current tokens for all mismatches in the batch
            context_tokens_list = []
            current_tokens_list = []
            
            for mismatch in batch_mismatches:
                # Store mismatch point for later reference
                mismatch_points_by_id[(mismatch.data_id, mismatch.token_id)] = mismatch
                data_mismatch_points[(mismatch.data_id, mismatch.token_id)] = mismatch
                
                # Add to batch input lists
                context_tokens_list.append(mismatch.context_tokens)
                current_tokens_list.append(mismatch.pred_small_token)
                
                if args.is_print:
                    print(f"\nMismatch (data_id={mismatch.data_id}, token_id={mismatch.token_id}):")
                    mismatch.print()

            # Small model generation - generate outputs for all mismatches in batch
            small_outputs_batch = gen_controller.generate_continuation(
                update_context_tokens=context_tokens_list,
                current_token=current_tokens_list,
                model_type='small',
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                num_samples=args.num_samples,
                top_p=args.top_p,
                top_k=args.top_k,
                num_continuation=args.num_continuation + 1
            )
            
            # Process the batch results
            # We need to match each output back to its corresponding mismatch
            output_idx = 0
            
            # Collect all diverge points for this batch
            all_batch_diverge_points = []
            
            for mismatch_idx, mismatch in enumerate(batch_mismatches):
                # Get this mismatch's outputs (accounting for num_samples)
                mismatch_outputs = small_outputs_batch[output_idx:output_idx + args.num_samples]
                output_idx += args.num_samples
                
                # Real model continuation (ground truth)
                # Use the real token as the reference token
                pred_comparison_token = mismatch.real_token
                current_idx = data_context.token_ids.index(mismatch.token_id)
                comparison_output = gen_controller.extract_real_continuation(
                    full_real_tokens=data_context.real_tokens,
                    current_token_index=current_idx,
                    max_new_tokens=args.max_new_tokens,
                    num_continuation=args.num_continuation + 1
                )
                pred_text = mismatch.real_text
                
                if not args.skip_stress_divergent_token:
                    pred_text = stressing_format(pred_text)
                
                # Get reference context
                common_context, comparison_diverge_context = gen_controller.get_latest_context(
                    context_text=mismatch.context_text,
                    pred_text=pred_text,
                    model_output_text=comparison_output["generated_text"],
                    is_next_context=(pred_comparison_token in gen_controller.stop_token_ids)
                    and (mismatch.prev_real_token in gen_controller.stop_token_ids),
                    num_continuation=args.num_continuation,
                    previous_context=args.previous_context,
                    common_previous_context=args.common_previous_context,
                )
                
                # Process each small model output for this mismatch
                mismatch_diverge_points = []
                for small_output in mismatch_outputs:
                    
                    pred_small_text = mismatch.pred_small_text
                    if not args.skip_stress_divergent_token:
                        pred_small_text = stressing_format(pred_small_text)
                        
                    # Get context for this small output
                    common_context, small_diverge_context = gen_controller.get_latest_context(
                        context_text=mismatch.context_text,
                        pred_text=pred_small_text,
                        model_output_text=small_output['generated_text'],
                        is_next_context=(mismatch.pred_small_token in gen_controller.stop_token_ids) 
                            and (mismatch.prev_real_token in gen_controller.stop_token_ids),
                        num_continuation=args.num_continuation,
                        previous_context=args.previous_context,
                        common_previous_context=args.common_previous_context,
                    )
                    
                    # Store the generation result
                    result = {
                        'data_id': mismatch.data_id,
                        'token_id': mismatch.token_id,
                        'small_diverge_text': small_diverge_context,
                        'reference_diverge_text': comparison_diverge_context,
                        'common_context': common_context
                    }
                    generation_results.append(result)
                    
                    # Create diverge points
                    diverge_point = DivergePoint(
                        data_id=mismatch.data_id,
                        token_id=mismatch.token_id,
                        small_diverge_text=small_diverge_context,
                        reference_diverge_text=comparison_diverge_context,
                        common_context=common_context,
                        pred_small_token=mismatch.pred_small_token,
                        pred_small_text=pred_small_text,
                    )
                    mismatch_diverge_points.append(diverge_point)
                
                # For non-verify case, process mismatch_diverge_points now
                
                    # Create comparison points without verify data
                for dp in mismatch_diverge_points:
                    # Create a ComparisonPoint with None/0 for verify fields
                    comparison_point = ComparisonPoint(
                        data_id=dp.data_id,
                        token_id=dp.token_id,
                        small_diverge_text=dp.small_diverge_text,
                        reference_diverge_text=dp.reference_diverge_text,
                        similarity_score=None,
                        verify_response=None,
                        common_context=dp.common_context,
                        pred_small_token=dp.pred_small_token,
                        pred_small_text=dp.pred_small_text,
                    )
                    data_comparison_points.append(comparison_point)
            
            # Log progress after each batch
            logger.info(f"Processed batch {batch_idx+1}/{num_batches} in data sample {data_id}")

        # Save results for this sentence
        data_df = data_points_to_df(data_comparison_points, data_mismatch_points, 'real', False)

        # Add data_id column if not present
        if 'data_id' not in data_df.columns:
            data_df['data_id'] = data_id

        # Handle first write in resume mode
        if not first_write_done:
            if args.resume and not existing_results_df.empty:
                # First write: save existing results first, then append new data
                logger.info(f"Resume mode: Writing {len(existing_results_df)} existing results first")
                existing_results_df.to_csv(results_path, index=False)
                data_df.to_csv(results_path, mode='a', header=False, index=False)
            else:
                # Normal first write
                data_df.to_csv(results_path, index=False)
            first_write_done = True
        else:
            # Subsequent writes: always append
            data_df.to_csv(results_path, mode='a', header=False, index=False)

        logger.info(f"Saved results for data sample {data_id}")

        comparison_points.extend(data_comparison_points)

    # Save final results
    results_df = data_points_to_df(comparison_points, mismatch_points_by_id, 'real', False)
    
    # In resume mode, combine existing results with new results
    if args.resume and not existing_results_df.empty:
        logger.info(f"Resume mode: Combining {len(existing_results_df)} existing results with {len(results_df)} new results")
        # Combine existing results with new results
        combined_results_df = pd.concat([existing_results_df, results_df], ignore_index=True)
        logger.info(f"Combined total: {len(combined_results_df)} results")
    else:
        combined_results_df = results_df
    
    # Save results with range in filename
    if args.low is not None and args.high is not None:
        final_results_filename = f"generation_results_data_{args.low}_to_{args.high}_real_full.csv"
    else:
        final_results_filename = f"generation_results_data_all_real_full.csv"

    final_results_path = output_path / final_results_filename
    logger.info(f"Saving final combined results to {final_results_path}")
    combined_results_df.to_csv(final_results_path, index=False)
    
    # Clean up resources
    print("Process completed.")
    logger.info("Cleaning up resources...")
    gen_controller.shutdown()
    logger.info("Done.")
    

if __name__ == "__main__":
    main()
