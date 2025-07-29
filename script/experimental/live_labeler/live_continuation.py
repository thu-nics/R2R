#!/usr/bin/env python3
import argparse
import torch
from typing import List, Tuple, Dict, Optional, Union
import time
import multiprocessing as mp
from tqdm import tqdm
from transformers import AutoTokenizer
import pandas as pd
import os
import json
from datasets import load_dataset, Dataset

from r2r.data.verify_model import ComparisonPoint
from r2r.models.sglang_wrapper import SGLangWrapper
from r2r.utils.config import MODEL_DICT, TOKEN
from r2r.utils.token_manager import SGLangTokenManager
from r2r.data.live_labeler import LiveDivergentLabeler
from r2r.evaluate.eval_utils import extract_boxed_answer


def launch_models(gpu_ids: List[int] = [0, 1], disable_cuda_graph: bool = True) -> Tuple[SGLangWrapper, SGLangWrapper]:
    """
    Launch the quick and reference language models.
    
    Args:
        gpu_ids: List of GPU IDs to use for quick and reference models
    
    Returns:
        Tuple of (quick_model, reference_model)
    """
    print(f"Initializing quick model: {MODEL_DICT['quick']['model_name']} on GPU {gpu_ids[0]}")

    quick_model = SGLangWrapper(
        model_path=MODEL_DICT['quick']['model_path'],
        gpu_id=gpu_ids[0],
        mem_fraction_static=MODEL_DICT['quick']['mem_fraction_static'],
        disable_cuda_graph=disable_cuda_graph,
        disable_overlap_schedule=True
    )
    
    print(f"Initializing reference model: {MODEL_DICT['reference']['model_name']} on GPU {gpu_ids[1]}")
    reference_model = SGLangWrapper(
        model_path=MODEL_DICT['reference']['model_path'],
        gpu_id=gpu_ids[1],
        mem_fraction_static=MODEL_DICT['reference']['mem_fraction_static'],
        disable_cuda_graph=disable_cuda_graph,
        disable_overlap_schedule=True
    )
    
    return quick_model, reference_model


def dummy_get_token_labels(
    context_tokens: List[List[int]],
    quick_tokens: List[int],
    ref_tokens: List[int]
) -> Tuple[List[int], List[int], List[Union[ComparisonPoint, None]]]:
    """
    Get the final tokens and model type indicators for a batch of tokens.
    
    Args:
        context_tokens: List of token contexts for each item in the batch
        quick_tokens: Tokens from quick model for each item in the batch
        ref_tokens: Tokens from reference model for each item in the batch
    
    Returns:
        Tuple of (final_tokens, token_types, comparison_points)
    """
    batch_size = len(context_tokens)
    final_tokens = []
    token_types = []
    comparison_points = []
    
    for i in range(batch_size):
        if quick_tokens[i] == ref_tokens[i]:
            final_tokens.append(ref_tokens[i])
            token_types.append(TOKEN.MATCH)
            comparison_points.append(None)
        else:
            final_tokens.append(ref_tokens[i])
            token_types.append(TOKEN.DIVERGENT)
            comparison_points.append(None)
            
    return final_tokens, token_types, comparison_points

def save_results(
    generated_tokens: List[List[int]],
    token_usage: List[List[int]],
    comparison_points: List[List[Union[ComparisonPoint, None]]],
    tokenizer,
    data_ids: List[str] = None,
    output_path: str = "live_continuation_results.csv",
    append: bool = False
):
    """
    Save the results of live continuation to a CSV file.
    
    Args:
        generated_tokens: List of generated token ID sequences
        token_usage: List of model usage indicators (0: same output, 1/2: different output)
        comparison_points: List of lists containing comparison points for each sequence
        tokenizer: Tokenizer used for decoding tokens
        data_ids: List of data IDs for each sequence
        output_path: Path to save the CSV file
        append: Whether to append to existing file or create a new one
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Prepare data for DataFrame
    data = []
    
    for seq_idx, (tokens, usage, seq_comparison_points) in enumerate(zip(generated_tokens, token_usage, comparison_points)):
        # Get the data ID if available
        data_id = data_ids[seq_idx] if data_ids and seq_idx < len(data_ids) else f"unknown_{seq_idx}"
        
        # Create a mapping from token position to comparison point for fast lookup
        point_map = {}
        for token_id, point in enumerate(seq_comparison_points):
            if point is not None:
                point_map[token_id] = point
        
        # Process each token in the sequence
        for pos, (token, usage_type) in enumerate(zip(tokens, usage)):
            # Base data for each token
            token_data = {
                "sequence_idx": seq_idx,
                "data_id": data_id,
                "position": pos,
                "token_id": token,
                "token_text": tokenizer.decode([token]),
                "usage_type": usage_type,
                "is_mismatch": usage_type != TOKEN.MATCH
            }
            
            # Add comparison point data if available
            if pos in point_map:
                point = point_map[pos]
                # Convert ComparisonPoint to dict and add relevant fields
                if hasattr(point, "__dict__"):
                    for key, value in point.__dict__.items():
                        if key not in ["pred_small_token", "data_id"]:  # Skip redundant fields
                            # Convert lists to strings for CSV compatibility
                            if isinstance(value, list):
                                token_data[f"{key}"] = str(value)
                            else:
                                token_data[f"{key}"] = value
                else:
                    # Handle if point is not a standard object with __dict__
                    token_data["point_data"] = str(point)
            
            data.append(token_data)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV, either append or create new
    if append and os.path.exists(output_path):
        # Append without header if file exists
        df.to_csv(output_path, index=False, mode='a', header=False)
        print(f"Results appended to {output_path}")
    else:
        # Create new file with header
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    
    # Update the summary file if creating new or appending
    if not append or not os.path.exists(output_path.replace('.csv', '_summary.json')):
        # Create a new summary
        summary = {
            "total_sequences": len(generated_tokens),
            "total_tokens": sum(len(tokens) for tokens in generated_tokens),
            "divergent_tokens": sum(usage.count(TOKEN.DIVERGENT) for usage in token_usage),
            "match_tokens": sum(usage.count(TOKEN.MATCH) for usage in token_usage),
            "match_percentage": sum(usage.count(TOKEN.MATCH) for usage in token_usage) / 
                               sum(len(usage) for usage in token_usage) * 100 if sum(len(usage) for usage in token_usage) > 0 else 0
        }
    else:
        # Update existing summary
        with open(output_path.replace('.csv', '_summary.json'), 'r') as f:
            summary = json.load(f)
            
        # Update with new data
        summary["total_sequences"] += len(generated_tokens)
        summary["total_tokens"] += sum(len(tokens) for tokens in generated_tokens)
        summary["divergent_tokens"] += sum(usage.count(TOKEN.DIVERGENT) for usage in token_usage)
        summary["match_tokens"] += sum(usage.count(TOKEN.MATCH) for usage in token_usage)
        
        # Recalculate match percentage
        if summary["total_tokens"] > 0:
            summary["match_percentage"] = (summary["match_tokens"] / summary["total_tokens"]) * 100
    
    # Save summary
    summary_path = output_path.replace('.csv', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    if not append:
        print(f"Summary saved to {summary_path}")
    else:
        print(f"Summary updated in {summary_path}")

def live_continuation(
    quick_model: SGLangWrapper,
    reference_model: SGLangWrapper,
    live_labeler: LiveDivergentLabeler,
    input_ids: List[List[int]],
    max_new_tokens: int = 100,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    repetition_penalty: Optional[float] = None
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Perform step-by-step continuation using both quick and reference models.
    If outputs differ, use the quick model's output.
    
    Args:
        quick_model: The small/fast model
        reference_model: The large/reference model
        input_ids: List of token ID sequences to continue
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Temperature for sampling
        top_p: Top-p probability threshold for nucleus sampling (optional)
        top_k: Top-k for sampling (optional)
        repetition_penalty: Penalty for token repetition (optional)
        
    Returns:
        Tuple of (
            List of generated token ID sequences,
            List of model usage indicators (0: same output, 1: different output, used quick model)
        )
    """
    batch_size = len(input_ids)
    tokenizer = quick_model.tokenizer
    
    # Initialize token manager with model type tracking
    token_manager = SGLangTokenManager(
        input_ids=input_ids,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        record_token_type=True
    )
    
    print(f"Starting generation for {batch_size} sequences with max {max_new_tokens} new tokens")
    
    position = 0
    
    # First step: Initialize with request objects
    active_input_ids = token_manager.get_active_input_ids()
    
    # Create initial requests for quick model
    quick_reqs = []
    for i, ids in enumerate(active_input_ids):
        req_id = f"quick_{position}_{i}"
        req = quick_model.create_req(
            rid=req_id,
            input_ids=ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty
        )
        quick_reqs.append(req)
    
    # Create initial requests for reference model
    ref_reqs = []
    for i, ids in enumerate(active_input_ids):
        req_id = f"ref_{position}_{i}"
        req = reference_model.create_req(
            rid=req_id,
            input_ids=ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty
        )
        ref_reqs.append(req)
        
    # Initial decode step with reqs
    quick_model.decode_step(quick_reqs)
    reference_model.decode_step(ref_reqs)
    
    # TODO: compare points do not support batch processing yet
    comparison_points_all = [[] for _ in range(batch_size)]
    final_tokens_all = [[] for _ in range(batch_size)]
    token_types_all = [[] for _ in range(batch_size)]

    # Use tqdm progress bar if available
    progress_bar = tqdm(range(max_new_tokens), desc="Generating tokens")

    last_batch_size = 0

    # Generate tokens until all sequences are complete or max tokens reached
    while not token_manager.is_generation_complete() and position < max_new_tokens:
        progress_bar.update(1)

        # Get active indices
        active_indices = token_manager.get_active_index()

        if len(active_indices) != last_batch_size:
            active_count = token_manager.get_active_count()
            print(f"Step {position}: {active_count}/{batch_size} sequences still active")
            last_batch_size = len(active_indices)
        
        if not active_indices:
            print(f"All sequences completed at step {position}")
            break
            
        # For subsequent steps, use continue generation signal (1)
        if position > 0:
            # Continue generation with current state for both models
            quick_model.decode_step(1)  # Signal to continue generation
            reference_model.decode_step(1)  # Signal to continue generation
        
        # Get results from both models
        quick_result = quick_model.decode_output_queue.get()
        ref_result = reference_model.decode_output_queue.get()
        
        # Handle potential errors
        if "error" in quick_result or "error" in ref_result:
            print(f"Error in decode step: {quick_result.get('error', '')} / {ref_result.get('error', '')}")
            break
            
        # Get the token IDs from both models as tensors (don't convert to lists)
        quick_next_tokens = quick_result["next_token_ids"]
        ref_next_tokens = ref_result["next_token_ids"]

        if not ((len(quick_next_tokens) == len(ref_next_tokens)) and len(quick_next_tokens) == len(active_indices)):
            print(f"Step {position}: {len(quick_next_tokens)}/{len(ref_next_tokens)}/{len(active_indices)}")
        
        contexts = token_manager.get_active_input_ids()

        # Process batch items in parallel
        quick_tokens = quick_next_tokens.tolist()
        ref_tokens = ref_next_tokens.tolist()

        # final_tokens, token_types, comparison_points = dummy_get_token_labels(
        #     contexts=contexts,
        #     quick_tokens=quick_tokens,
        #     ref_tokens=ref_tokens
        # )
        final_tokens, token_types, comparison_points = live_labeler.get_token_labels(
            contexts=contexts,
            quick_tokens=quick_tokens,
            ref_tokens=ref_tokens
        )
        
        # Modify the comparison points, final tokens, and token types
        for i, index in enumerate(active_indices):
            # record comparison points
            comparison_point = comparison_points[i]
            comparison_points_all[active_indices[i]].append(comparison_point)

            # apply final token
            final_token = final_tokens[i]
            quick_next_tokens[i] = final_token
            ref_next_tokens[i] = final_token
        
        # Update both models with their respective choices (as tensors, not lists)
        quick_model.update_step(quick_next_tokens)
        reference_model.update_step(ref_next_tokens)
        
        # Wait for update confirmation
        quick_model.update_output_queue.get()
        reference_model.update_output_queue.get()
        
        # Record generated tokens and token types
        token_manager.update_sequences_direct(
            generated_tokens=final_tokens, 
            token_types=token_types
        )
        
        # Increment position
        position += 1
        
    # Clean up all unfinished requests
    quick_model.clean_up()
    reference_model.clean_up()

    # Get final outputs from token manager
    final_outputs = token_manager.get_final_outputs()
    
    # Calculate total tokens generated
    total_tokens_generated = sum(len(output["output_ids"]) for output in final_outputs)
    
    # Extract generated token IDs and token types
    generated_tokens = []
    token_usage = []
    
    for output in final_outputs:
        generated_tokens.append(output["output_ids"])
        token_usage.append(output["token_types"])

    # Calculate token match statistics
    total_matches = sum(usage.count(TOKEN.MATCH) for usage in token_usage)
    total_tokens = sum(len(usage) for usage in token_usage)
    match_percentage = (total_matches / total_tokens * 100) if total_tokens > 0 else 0
    
    print(f"Token matches: {total_matches}/{total_tokens} ({match_percentage:.2f}%)")
    
    return generated_tokens, token_usage, comparison_points_all

def load_problems(
    problem_ids: List[str], 
    dataset_path: str, 
    question_key: str
) -> Tuple[List[str], List[str]]:
    """
    Load problems from a specified Hugging Face dataset using a list of problem IDs.
    
    Args:
        problem_ids: List of problem IDs to load.
        dataset_id: Hugging Face dataset identifier (e.g., "user/dataset_name").
        dataset_key: The key in the dataset corresponding to the problem text.
        
    Returns:
        Tuple of (list of problem texts, list of successfully loaded problem IDs)
    """
    if not problem_ids:
        print("Warning: No problem IDs provided.")
        return [], []

    try:
        # Load the specified HuggingFace dataset
        print(f"Loading dataset: {dataset_path}")
        dataset = load_dataset(dataset_path, split='train') # Assuming 'train' split
    except Exception as e:
        print(f"Error: Could not load dataset {dataset_path}: {e}")
        return [], []

    problems = []
    loaded_ids = []
    
    for problem_id in problem_ids:
        # Assumes the dataset has a column named 'ID' for matching problem_id
        matching_entries = dataset.filter(lambda example: example.get('ID') == problem_id)
        
        if matching_entries:
            if len(matching_entries) > 1:
                print(f"Warning: Multiple entries found for ID {problem_id} in dataset {dataset_path}. Using the first one.")
            
            problem_text = matching_entries[0].get(question_key)
            if problem_text is not None:
                problems.append(problem_text)
                loaded_ids.append(problem_id)
            else:
                print(f"Warning: Found entry for ID {problem_id} in {dataset_path}, but key '{question_key}' is missing or null.")
        else:
            print(f"Warning: Problem with ID {problem_id} not found in dataset {dataset_path}.")

    print(f"Successfully loaded {len(problems)} problems for {len(loaded_ids)} IDs from dataset {dataset_path}.")
    return problems, loaded_ids
    


def create_sequence_summary(input_csv_path: str, output_csv_path: str):
    """
    Processes a detailed live continuation results CSV to create a summary CSV 
    with statistics for each sequence.

    Args:
        input_csv_path: Path to the detailed CSV file generated by save_results.
        output_csv_path: Path to save the summary CSV file.
    """
    if not os.path.exists(input_csv_path):
        print(f"Error: Input CSV file not found: {input_csv_path}")
        return

    try:
        df = pd.read_csv(input_csv_path)
    except Exception as e:
        print(f"Error reading input CSV {input_csv_path}: {e}")
        return

    # Ensure correct sorting for text reconstruction
    df = df.sort_values(by=['data_id', 'position'])

    # Group by sequence ID and aggregate
    summary_data = []
    for data_id, group in df.groupby('data_id'):
        model_output = "".join(group['token_text'].astype(str))
        mismatch_rate = group['is_mismatch'].mean()
        # Assuming TOKEN.DIVERGENT == 1 based on typical usage
        divergent_rate = (group['usage_type'] == TOKEN.DIVERGENT).mean() 
        output_tokens = len(group)  # Count of rows (tokens) for this data_id
        
        # Calculate model_params using formula: SLM_params + divergent_rate * LLM_params
        slm_params = float(MODEL_DICT['quick']['param'])  # Small Language Model params
        llm_params = float(MODEL_DICT['reference']['param'])  # Large Language Model params
        model_params = slm_params + divergent_rate * llm_params
        
        # Extract predicted answer from model output
        predicted_answer, has_extracted_answer = extract_boxed_answer(model_output)
        
        summary_data.append({
            "problem_id": data_id,
            "model_name": "live_continuation",  # Could be made configurable
            "has_extracted_answer": has_extracted_answer,
            "predicted_answer": predicted_answer,
            "output_tokens": output_tokens,
            "total_tokens": "",  # Would need input_tokens to calculate
            "run_time": "",  # Would need to be tracked separately
            "model_params": model_params,
            "model_output": model_output,
            "mismatch_rate": mismatch_rate,
            "divergent_rate": divergent_rate
        })

    if not summary_data:
        print(f"No data found to summarize in {input_csv_path}")
        return

    summary_df = pd.DataFrame(summary_data)

    # Save the summary CSV
    try:
        summary_df.to_csv(output_csv_path, index=False)
        print(f"Sequence summary saved to: {output_csv_path}")
    except Exception as e:
        print(f"Error writing summary CSV {output_csv_path}: {e}")


def convert_intermediate_to_summary(input_csv_path: str, output_csv_path: str = None):
    """
    Convert intermediate live continuation results to a summary CSV.
    This function reuses the existing create_sequence_summary function.
    
    Args:
        input_csv_path: Path to the intermediate results CSV file
        output_csv_path: Path to save the summary CSV file. If None, will be auto-generated
    """
    if not os.path.exists(input_csv_path):
        print(f"Error: Input CSV file not found: {input_csv_path}")
        return
    
    # Auto-generate output path if not provided
    if output_csv_path is None:
        input_dir = os.path.dirname(input_csv_path)
        input_filename = os.path.basename(input_csv_path)
        
        # Extract timestamp from filename if present
        if "live_continuation_results_" in input_filename:
            timestamp_part = input_filename.replace("live_continuation_results_", "").replace(".csv", "")
            output_filename = f"live_continuation_summary_{timestamp_part}.csv"
        else:
            # Use current timestamp if no timestamp found in filename
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            output_filename = f"live_continuation_summary_{timestamp}.csv"
        
        output_csv_path = os.path.join(input_dir, output_filename)
    
    print(f"Converting intermediate results from: {input_csv_path}")
    print(f"Summary will be saved to: {output_csv_path}")
    
    # Reuse the existing create_sequence_summary function
    create_sequence_summary(input_csv_path, output_csv_path)
    
    print(f"Conversion completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Live continuation with quick and reference models")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0, 1], help="GPU IDs to use")
    parser.add_argument("--max_tokens", type=int, default=8192, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p probability threshold")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k for sampling")
    parser.add_argument("--repetition_penalty", type=float, default=None, help="Penalty for token repetition")
    parser.add_argument("--output_dir", type=str, default="output/live_continuation", help="Directory to save the results")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing texts")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of texts to process")
    parser.add_argument("--convert_to_summary", type=str, help="Convert intermediate results CSV to summary CSV")
    parser.add_argument("--summary_output", type=str, help="Output path for summary CSV (used with --convert_to_summary)")
    args = parser.parse_args()
    
    # Handle conversion mode
    if args.convert_to_summary:
        convert_intermediate_to_summary(args.convert_to_summary, args.summary_output)
        return
    
    # Create timestamp for filename
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    # Define path for the detailed results CSV
    detailed_output_path = os.path.join(args.output_dir, f"live_continuation_results_{timestamp}.csv")
    # Define path for the sequence summary CSV
    summary_output_path = os.path.join(args.output_dir, f"live_continuation_summary_{timestamp}.csv")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load all texts
    example_problem_ids = [
        "2024-I-4",
        "2024-II-4",
        "2024-II-6",
        "2024-I-1",
        "2024-I-3",
        "2024-II-10",
        "2024-II-12",
        "2024-I-7",
        "2024-I-6",
        "2024-I-2",
        "2024-I-14",
        "2024-II-3",
        "2024-I-15",
        "2024-I-9",
        "2024-II-7",
        "2024-I-13",
        "2024-II-13"
    ]


    example_dataset_path = "Maxwell-Jia/AIME_2024" # Example dataset
    example_dataset_key = "Problem" # Example key for problem text

    queries, ids = load_problems(
        problem_ids=example_problem_ids,
        dataset_path=example_dataset_path,
        question_key=example_dataset_key
    )
    if args.limit:
        queries = queries[:args.limit]
        ids = ids[:args.limit]

    print(f"Total texts loaded: {len(queries)}")
    print(f"Loaded ids: {ids}")

    disable_cuda_graph = False

    # Launch models
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DICT['quick']['model_path'])
    live_labeler = LiveDivergentLabeler(
        tokenizer=tokenizer,
        verify_model_name=MODEL_DICT['verify']['model_path'],
        verify_mode="common_context",
        continuation_max_new_tokens=128,
        verifier_max_new_tokens=16,
        num_samples=1,
        num_continuation=1,
        previous_context=0
    )
    quick_model, reference_model = launch_models(args.gpu_ids, disable_cuda_graph=disable_cuda_graph)
    
    # Process texts in batches
    batch_size = args.batch_size
    
    for i in range(0, len(queries), batch_size):
        batch_texts = queries[i:i+batch_size]
        batch_ids = ids[i:i+batch_size] if ids else None
        print(f"\nProcessing batch {i//batch_size + 1}/{(len(queries) + batch_size - 1)//batch_size}")
        print(f"Batch size: {len(batch_texts)}")
        
        # Tokenize batch
        prompts = quick_model.text_to_prompt(batch_texts)
        input_ids = [quick_model.tokenizer.encode(prompt) for prompt in prompts]
        
        # Run live continuation for this batch
        start_time = time.time()
        generated_tokens, token_usage, comparison_points = live_continuation(
            quick_model=quick_model,
            reference_model=reference_model,
            live_labeler=live_labeler,
            input_ids=input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty
        )
        elapsed = time.time() - start_time
        
        output_texts = tokenizer.batch_decode(generated_tokens)
        for output_text in output_texts:
            print(output_text)
        print(generated_tokens)
        print(token_usage)
        
        # Save results for this batch (appending if not the first batch)
        save_results(
            generated_tokens=generated_tokens,
            token_usage=token_usage,
            comparison_points=comparison_points,
            tokenizer=tokenizer,
            data_ids=batch_ids,
            output_path=detailed_output_path,
            append=(i > 0)  # Only append if not the first batch
        )
        
        # Print batch results
        print(f"\nBatch {i//batch_size + 1} completed in {elapsed:.2f} seconds")
        for j, (prompt, tokens, usage) in enumerate(zip(prompts, generated_tokens, token_usage)):
            output = quick_model.tokenizer.decode(tokens)
            agreement_rate = (len(usage) - sum(usage)) / len(usage) if usage else 1.0
            print(f"\nText {i+j+1}: {prompt[:50]}...")
            print(f"Output: {output[:50]}...")
            print(f"Model agreement rate: {agreement_rate:.2%}")
    
    print(f"\nAll batches processed. Total texts: {len(queries)}")
    print(f"Detailed results saved to: {detailed_output_path}")

    # Create the sequence summary CSV after all batches are done
    create_sequence_summary(detailed_output_path, summary_output_path)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
