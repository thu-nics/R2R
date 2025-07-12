"""
Step2 generates the LLM continuation based on the response prefix and the non-identical SLM-token predictions.

Inputs:
- Prediction comparison CSV file (specified by `--input_path`) from step 1. This file contains mismatch points between SLM and LLM tokens.

Outputs:
- A directory (specified by `--output_path`, default: `output/playground/continuation`) containing:
    - `args.json`: A JSON file logging the arguments used for the script execution.
    - `generation_results_data_all_real.csv`: Each row corresponds to a processed mismatch, detailing the generated continuation from the SLM token and the LLM continuation, along with their respective contexts.

Resume Functionality:
- Use `--resume` flag to skip already processed data IDs by checking existing output CSV files.
- Works with both range-specific (--low/--high) and full dataset processing.
- Automatically detects and skips data IDs that have already been saved to the output file.
"""

from pathlib import Path
import pandas as pd
import logging
from tqdm import tqdm
import argparse
import json
import concurrent.futures
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from r2r.data.data_process import DataProcessor, MismatchPoint
from r2r.data.generation_controller import ModelController, DivergePoint
from r2r.data.verify_model import ComparisonPoint, data_points_to_df

def parse_args():
    parser = argparse.ArgumentParser(description='Process data with specified data sample ID range')
    parser.add_argument('--input_path', type=str, default="output/playground/prediction_comparison.csv", help='Path to input CSV file. If not provided, will use default path.')
    parser.add_argument('--output_path', type=str, default='output/playground/continuation', help='Path to store output CSV files')
    parser.add_argument('--api_url_main', type=str, default='http://localhost:30000', help='API URL for the main model')
    parser.add_argument('--api_url_reference', type=str, default='http://localhost:30000', help='API URL for the reference model')
    parser.add_argument('--request_timeout', type=int, default=6000, help='Timeout for API requests in seconds')
    parser.add_argument('--low', type=int, default=None, help='Lower bound of data sample ID range (inclusive). If not specified, will process all data.')
    parser.add_argument('--high', type=int, default=None, help='Upper bound of data sample ID range (exclusive). If not specified, will process all data.')
    parser.add_argument('--max_tokens', type=int, default=32768, help='Maximum number of tokens per data sample')
    parser.add_argument('--max_new_tokens', type=int, default=8192, help='Maximum number of new tokens to generate')
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
    parser.add_argument('--max_concurrent_requests', type=int, default=32,
                        help='Maximum number of concurrent requests to send to the model')
    parser.add_argument('--skip_stress_divergent_token', action='store_true', default=False,
                        help='Whether to skip stressing the divergent token')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Whether to resume from existing results by skipping already processed data IDs')
    parser.add_argument('--is_print', action='store_true', default=False,
                        help='Whether to print the results')
    return parser.parse_args()

def get_processed_data_ids(results_path):
    """Read existing results file and return set of processed data_ids"""
    if not results_path.exists():
        return set()
    
    try:
        existing_df = pd.read_csv(results_path)
        if 'data_id' in existing_df.columns:
            processed_ids = set(existing_df['data_id'].unique())
            logger.info(f"Found {len(processed_ids)} already processed data IDs in {results_path}")
            return processed_ids
        else:
            logger.warning(f"No 'data_id' column found in existing results file {results_path}")
            return set()
    except Exception as e:
        logger.warning(f"Could not read existing results file {results_path}: {e}")
        return set()

def process_single_mismatch(args, gen_controller, mismatch, data_contexts, stressing_format):
    """Process a single mismatch and return comparison points."""
    try:
        # Get the data context for this mismatch's data_id
        data_context = data_contexts[mismatch.data_id]
        
        # Store mismatch point for later reference
        data_mismatch_points = {(mismatch.data_id, mismatch.token_id): mismatch}
        
        # Small model generation - generate outputs for this mismatch
        small_outputs = gen_controller.generate_continuation_single(
            update_context_tokens=mismatch.context_tokens,
            current_token=mismatch.pred_small_token,
            model_type='small',
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            num_samples=args.num_samples,
            top_p=args.top_p,
            top_k=args.top_k,
            num_continuation=args.num_continuation + 1
        )
        
        # Real model continuation (ground truth)
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
        mismatch_comparison_points = []
        for small_output in small_outputs:
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
            
            # Create diverge point
            diverge_point = DivergePoint(
                data_id=mismatch.data_id,
                token_id=mismatch.token_id,
                small_diverge_text=small_diverge_context,
                reference_diverge_text=comparison_diverge_context,
                common_context=common_context,
                pred_small_token=mismatch.pred_small_token,
                pred_small_text=pred_small_text,
            )
            
            # Create comparison point without verify data
            comparison_point = ComparisonPoint(
                data_id=diverge_point.data_id,
                token_id=diverge_point.token_id,
                small_diverge_text=diverge_point.small_diverge_text,
                reference_diverge_text=diverge_point.reference_diverge_text,
                similarity_score=None,
                verify_response=None,
                common_context=diverge_point.common_context,
                pred_small_token=diverge_point.pred_small_token,
                pred_small_text=diverge_point.pred_small_text,
            )
            mismatch_comparison_points.append(comparison_point)
            
        return mismatch_comparison_points, data_mismatch_points
        
    except Exception as e:
        logger.error(f"Error processing mismatch (data_id={mismatch.data_id}, token_id={mismatch.token_id}): {str(e)}")
        return [], {}

def batch_concurrent_mismatch_processing(args, gen_controller, all_mismatches, data_contexts, stressing_format, max_concurrent=10):
    """Process all mismatches from all data samples concurrently in batches."""
    all_comparison_points = []
    all_mismatch_points = {}
    
    # Create a thread pool executor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        # Submit all tasks
        future_to_mismatch = {}
        for mismatch in all_mismatches:
            future = executor.submit(process_single_mismatch, args, gen_controller, mismatch, data_contexts, stressing_format)
            future_to_mismatch[future] = mismatch
        
        # Process completed requests
        for future in tqdm(concurrent.futures.as_completed(future_to_mismatch), total=len(future_to_mismatch), desc="Processing all mismatches"):
            mismatch = future_to_mismatch[future]
            try:
                comparison_points, mismatch_points = future.result()
                all_comparison_points.extend(comparison_points)
                all_mismatch_points.update(mismatch_points)
                
                if args.is_print:
                    print(f"\nMismatch (data_id={mismatch.data_id}, token_id={mismatch.token_id}):")
                    mismatch.print()
                    
            except Exception as e:
                logger.error(f"Error processing mismatch {mismatch.data_id}-{mismatch.token_id}: {str(e)}")
    
    return all_comparison_points, all_mismatch_points

def main():
    args = parse_args()
    
    # Use input_path from args if provided, otherwise use default
    input_path = Path(args.input_path)

    # Use output_path from args
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    # convert args to json
    args_json = args.__dict__
    with open(output_path / 'args.json', 'w') as f:
        json.dump(args_json, f)
    # Define output path at the start
    if args.low is not None and args.high is not None:
        results_filename = f"generation_results_data_{args.low}_to_{args.high}_real.csv"
    else:
        results_filename = f"generation_results_data_all_real.csv"

    results_path = output_path / results_filename
    logger.info(f"Results will be saved to {results_path}")
    logger.info(f"Using real tokens (ground truth) as comparison model")
    logger.info(f"Using max concurrent requests of {args.max_concurrent_requests} for generation")
    
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
    if args.resume:
        logger.info("Resume mode enabled - checking for existing results...")
        processed_data_ids = get_processed_data_ids(results_path)
        
        if processed_data_ids:
            # Filter out already processed data IDs
            original_count = len(sample_data)
            sample_data = sample_data[~sample_data['data_id'].isin(processed_data_ids)]
            remaining_count = len(sample_data)
            
            logger.info(f"Resume: Filtered out {original_count - remaining_count} already processed data IDs")
            logger.info(f"Resume: {remaining_count} data samples remaining to process")
            
            if remaining_count == 0:
                logger.info("All data samples have already been processed. Exiting.")
                return
        else:
            logger.info("Resume: No existing results found, processing all data samples")
    else:
        logger.info("Resume mode disabled - processing all data samples (existing results may be overwritten)")

    # Process the data
    processor = DataProcessor(sample_data, max_tokens=args.max_tokens, comparison_model='real')

    # Initialize the generation controller and verify model
    gen_controller = ModelController(
        mode='api',
        comparison_model='real', 
        api_url_main=args.api_url_main, 
        api_url_reference=args.api_url_reference, 
        request_timeout=args.request_timeout
    )
    
    logger.info("verify model disabled (--verify flag not provided)")
        
    # Test group_mismatches_by_data_id
    print("\nTesting group_mismatches_by_data_id:")
    grouped_mismatches = processor.group_mismatches_by_data_id()

    # Collect all mismatches and data contexts for batch processing
    all_mismatches = []
    data_contexts = {}
    total_mismatches = 0
    
    logger.info("Collecting all mismatches and data contexts...")
    for data_id, mismatches in grouped_mismatches.items():
        # Get the data context for this data_id
        data_context = processor.get_data_context(data_id)
        data_contexts[data_id] = data_context
        
        # Add all mismatches to the batch list
        all_mismatches.extend(mismatches)
        total_mismatches += len(mismatches)
        
        logger.info(f"Data sample {data_id}: {len(mismatches)} mismatches")
    
    logger.info(f"Total mismatches to process: {total_mismatches}")

    # Dictionary to store mismatch points by (data_id, token_id) for later reference
    mismatch_points_by_id = {}
    for mismatch in all_mismatches:
        mismatch_points_by_id[(mismatch.data_id, mismatch.token_id)] = mismatch

    # Process all mismatches concurrently in batch
    logger.info("Starting batch processing of all mismatches...")
    comparison_points, all_mismatch_points = batch_concurrent_mismatch_processing(
        args=args,
        gen_controller=gen_controller,
        all_mismatches=all_mismatches,
        data_contexts=data_contexts,
        stressing_format=stressing_format,
        max_concurrent=args.max_concurrent_requests
    )
    
    logger.info(f"Batch processing completed. Processed {len(comparison_points)} comparison points.")

    # Update mismatch_points_by_id with concurrent results
    mismatch_points_by_id.update(all_mismatch_points)

    # Save results immediately after batch processing
    logger.info("Saving batch results...")
    results_df = data_points_to_df(comparison_points, mismatch_points_by_id, 'real', False)
    
    # Save results with range in filename
    if args.low is not None and args.high is not None:
        batch_results_filename = f"generation_results_data_{args.low}_to_{args.high}_real_batch.csv"
    else:
        batch_results_filename = f"generation_results_data_all_real_batch.csv"

    batch_results_path = output_path / batch_results_filename
    results_df.to_csv(batch_results_path, index=False)
    logger.info(f"Batch results saved to {batch_results_path}")

    # Also save in incremental format (compatible with original format)
    # Group results by data_id for incremental saving
    results_by_data_id = {}
    for point in comparison_points:
        data_id = point.data_id
        if data_id not in results_by_data_id:
            results_by_data_id[data_id] = []
        results_by_data_id[data_id].append(point)

    # Save incremental results for each data_id (for compatibility)
    for data_id, data_comparison_points in results_by_data_id.items():
        # Get mismatch points for this data_id
        data_mismatch_points = {
            key: value for key, value in mismatch_points_by_id.items() 
            if key[0] == data_id
        }
        
        # Create dataframe for this data_id
        data_df = data_points_to_df(data_comparison_points, data_mismatch_points, 'real', False)

        # Add data_id column if not present
        if 'data_id' not in data_df.columns:
            data_df['data_id'] = data_id

        # Save in append mode if file exists, otherwise create new file
        if results_path.exists():
            data_df.to_csv(results_path, mode='a', header=False, index=False)
        else:
            data_df.to_csv(results_path, index=False)

    logger.info(f"Incremental results saved to {results_path}")

    # Update comparison_points for final processing (already contains all points)

    # Final results already saved as batch results
    logger.info(f"All processing completed. Total comparison points: {len(comparison_points)}")
    
    # Clean up resources
    print("Process completed.")
    logger.info("Cleaning up resources...")
    gen_controller.shutdown()
    logger.info("Done.")
    

if __name__ == "__main__":
    main()
