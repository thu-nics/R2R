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
from r2r.data.verify_model import VerifyModel, ComparisonPoint, data_points_to_df

def parse_args():
    parser = argparse.ArgumentParser(description='Process data with specified data sample ID range')
    parser.add_argument('--input_path', type=str, default="output/playground/prediction_comparison.csv", help='Path to input CSV file. If not provided, will use default path.')
    parser.add_argument('--output_path', type=str, default='output/playground/continuation', help='Path to store output CSV files')

    parser.add_argument('--tp_size', type=int, default=2, help='Number of tensor parallel GPUs')
    parser.add_argument('--dp_size', type=int, default=1, help='Number of data parallel GPUs')
    parser.add_argument('--low', type=int, default=None, help='Lower bound of data sample ID range (inclusive). If not specified, will process all data.')
    parser.add_argument('--high', type=int, default=None, help='Upper bound of data sample ID range (exclusive). If not specified, will process all data.')
    parser.add_argument('--max_tokens', type=int, default=8192, help='Maximum number of tokens per data sample')
    parser.add_argument('--max_new_tokens', type=int, default=8192, help='Maximum number of new tokens to generate')
    parser.add_argument('--gen_mem_fraction', type=float, default=0.9,
                        help='Fraction of GPU memory to allocate for generation')
    parser.add_argument('--verify_mem_fraction', type=float, default=0.3,
                        help='Fraction of GPU memory to allocate for judging')
    parser.add_argument('--num_continuation', type=int, default=-1, 
                        help='Number of continuations to generate (number of times to encounter EOS tokens)')
    parser.add_argument('--previous_context', type=int, default=-1,
                        help='Number of previous sentences to include in the context')
    parser.add_argument('--common_previous_context', type=int, default=-1,
                        help='Number of previous sentences to include in the common context')
    parser.add_argument('--verify_model', type=str, default='Qwen/Qwen2.5-1.5B-Instruct',
                        help='Choose which model to use for judging, default is "Qwen/Qwen2.5-1.5B-Instruct"')
    parser.add_argument('--verify_mode', type=str, default='divergent', choices=['similarity', 'divergent', 'common_context'],
                        help='Choose the verify mode: "similarity" (1-10 scale) or "divergent" (0-1 binary score) or "common_context" (0-1 binary score)')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of small model outputs to generate for each mismatch point')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Temperature for sampling. Higher values produce more random samples.')
    parser.add_argument('--top_p', type=float, default=1.0,
                        help='Top-p probability threshold for nucleus sampling (0 < top_p ≤ 1)')
    parser.add_argument('--is_print', action='store_true',
                        help='Whether to print the results')
    parser.add_argument('--is_verify', action='store_true',
                        help='Whether to initialize and use the verify model for evaluations')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of mismatches to batch together for generation')
    parser.add_argument('--stress_divergent_token',action='store_true',
                        help='Whether to stress the divergent token')
    return parser.parse_args()

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
        if args.is_verify:
            results_filename = f"comparison_results_data_{args.low}_to_{args.high}_real_{args.verify_mode}.csv"
        else:
            results_filename = f"generation_results_data_{args.low}_to_{args.high}_real.csv"
    else:
        if args.is_verify:
            results_filename = f"comparison_results_data_all_real_{args.verify_mode}.csv"
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

    # Process the data
    processor = DataProcessor(sample_data, max_tokens=args.max_tokens, comparison_model='real')

    # Initialize the generation controller and verify model
    gen_controller = ModelController(comparison_model='real', mem_fraction_static=args.gen_mem_fraction, tp_size=args.tp_size, dp_size=args.dp_size)
    
    # Only initialize verify model if --verify flag is provided
    verify_model = None
    if args.is_verify:
        verify_model = VerifyModel(model_name=args.verify_model, verify_mode=args.verify_mode, mem_fraction_static=args.verify_mem_fraction, tp_size=args.tp_size)
        # Log verify model mode
        logger.info(f"Using {args.verify_mode} mode for verify model")
    else:
        logger.info("verify model disabled (--verify flag not provided)")
        
    # Test group_mismatches_by_data_id
    print("\nTesting group_mismatches_by_data_id:")
    grouped_mismatches = processor.group_mismatches_by_data_id()

    # Dictionary to store mismatch points by (data_id, token_id) for later reference
    mismatch_points_by_id = {}

    # Process all sentences and collect comparison points
    comparison_points = []
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
                
                if args.stress_divergent_token:
                    pred_text = stressing_format(pred_text)
                
                # Get reference context
                common_context, comparison_diverge_context = gen_controller.get_latest_context(
                    context_text=mismatch.context_text,
                    pred_text=pred_text,
                    model_output_text=comparison_output["generated_text"],
                    is_next_context=(pred_comparison_token in gen_controller.eos_token_ids)
                    and (mismatch.prev_real_token in gen_controller.eos_token_ids),
                    num_continuation=args.num_continuation,
                    previous_context=args.previous_context,
                    common_previous_context=args.common_previous_context,
                )
                
                # Process each small model output for this mismatch
                mismatch_diverge_points = []
                for small_output in mismatch_outputs:
                    
                    pred_small_text = mismatch.pred_small_text
                    if args.stress_divergent_token:
                        pred_small_text = stressing_format(pred_small_text)
                        
                    # Get context for this small output
                    common_context, small_diverge_context = gen_controller.get_latest_context(
                        context_text=mismatch.context_text,
                        pred_text=pred_small_text,
                        model_output_text=small_output['generated_text'],
                        is_next_context=(mismatch.pred_small_token in gen_controller.eos_token_ids) 
                            and (mismatch.prev_real_token in gen_controller.eos_token_ids),
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
                    
                    # Add to batch collection for verify model
                    if args.is_verify:
                        all_batch_diverge_points.append(diverge_point)
                
                # For non-verify case, process mismatch_diverge_points now
                if not args.is_verify:
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
            
            # Process all diverge points for this batch with verify model
            if args.is_verify and all_batch_diverge_points:
                # Batch compare all the divergent continuations for this batch at once
                logger.info(f"Processing {len(all_batch_diverge_points)} diverge points with verify model (batch size = {args.batch_size}*{args.num_samples})")
                batch_comparison_points = verify_model.batch_compare_diverge_points(all_batch_diverge_points)
                
                if args.is_print:
                    logger.info(f"verify model completed processing {len(all_batch_diverge_points)} diverge points")
                    for i, comparison_point in enumerate(batch_comparison_points[:5]):  # Print just the first few
                        comparison_point.print()
                
                # Add all comparison points
                data_comparison_points.extend(batch_comparison_points)
            
            # Log progress after each batch
            logger.info(f"Processed batch {batch_idx+1}/{num_batches} in data sample {data_id}")

        # Save results for this sentence
        data_df = data_points_to_df(data_comparison_points, data_mismatch_points, 'real', args.is_verify)

        # Add data_id column if not present
        if 'data_id' not in data_df.columns:
            data_df['data_id'] = data_id

        # Save in append mode if file exists, otherwise create new file
        if results_path.exists():
            data_df.to_csv(results_path, mode='a', header=False, index=False)
        else:
            data_df.to_csv(results_path, index=False)

        logger.info(f"Saved results for data sample {data_id}")

        comparison_points.extend(data_comparison_points)

    # Save final results
    results_df = data_points_to_df(comparison_points, mismatch_points_by_id, 'real', args.is_verify)
    
    # Save results with range in filename
    if args.low is not None and args.high is not None:
        if args.is_verify:
            final_results_filename = f"comparison_results_data_{args.low}_to_{args.high}_real_{args.verify_mode}_full.csv"
        else:
            final_results_filename = f"generation_results_data_{args.low}_to_{args.high}_real_full.csv"
    else:
        if args.is_verify:
            final_results_filename = f"comparison_results_data_all_real_{args.verify_mode}_full.csv"
        else:
            final_results_filename = f"generation_results_data_all_real_full.csv"

    results_path = output_path / final_results_filename
    logger.info(f"Saving results to {results_path}")
    results_df.to_csv(results_path, index=False)

    # Print summary statistics if needed
    if args.is_print and args.is_verify:
        # Only print verify statistics if verify model was used
        # Read the complete results file for statistics
        final_results = pd.read_csv(results_path)
        print("\nSummary Statistics:")
        print(f"Total comparisons: {len(final_results)}")

        if args.verify_mode == "similarity":
            # For similarity mode (1-10 scale)
            print("\nSimilarity Score Distribution:")
            similarity_dist = final_results['similarity_score'].value_counts().sort_index()
            print(similarity_dist.to_frame(name='count'))
            print(f"\nAverage Similarity Score: {final_results['similarity_score'].mean():.2f}/10")
        else:
            # For divergent mode (0-1 binary)
            print("\nDivergent Score Distribution:")
            divergent_dist = final_results['similarity_score'].value_counts().sort_index()
            print(divergent_dist.to_frame(name='count'))
            if 0 in divergent_dist.index and 1 in divergent_dist.index:
                same_percentage = (divergent_dist[0] / len(final_results)) * 100
                different_percentage = (divergent_dist[1] / len(final_results)) * 100
                print(f"\nPercentage of Same (0): {same_percentage:.2f}%")
                print(f"Percentage of Different (1): {different_percentage:.2f}%")
    elif not args.is_verify:
        logger.info("verify model disabled, skipping comparison statistics")
    
    # Clean up resources
    print("Process completed.")
    logger.info("Cleaning up resources...")
    gen_controller.shutdown()
    if verify_model:
        verify_model.shutdown()
    logger.info("Done.")
    

if __name__ == "__main__":
    main()
