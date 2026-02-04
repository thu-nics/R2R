import os
os.environ["SGLANG_ENABLE_TORCH_COMPILE"] = "0"

import sys
import time
import argparse
import torch
import multiprocessing as mp
import warnings
import csv
from datetime import datetime

from r2r.models.sglang_patch.sl_disaggregation_system import SLDisaggregationSystem
import json


# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ["SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
torch.set_warn_always(False)


def save_results_to_csv(csv_path: str, results: dict, append: bool = True):
    """Save results to CSV file."""
    file_exists = os.path.isfile(csv_path)
    mode = 'a' if append else 'w'
    
    fieldnames = [
        'timestamp', 'system', 'config_path', 'threshold', 'batch_size',
        'input_tokens', 'output_tokens', 'total_time_s', 
        'decode_throughput_tps', 'quick_ratio', 'reference_ratio',
        'prompt', 'generated_text'
    ]
    
    with open(csv_path, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists or not append:
            writer.writeheader()
        writer.writerow(results)
    
    print(f"Results saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Test SLDisaggregationSystem - Speed Test")
    parser.add_argument('--config-path', type=str, 
                        default="config/DeepSeek-R1-Distill-Qwen-1.5B+DeepSeek-R1-Distill-Qwen-32B_local.yaml",
                        help='Path to config.yaml')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for neural routing (fallback if not in config)')
    
    # Test parameters
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for generation')
    parser.add_argument('--decode-length', type=int, default=8192,
                        help='Maximum number of tokens to decode/generate')
    parser.add_argument('--slm-tp-size', type=int, default=1,
                        help='TP size for SLM')
    parser.add_argument('--llm-tp-size', type=int, default=1,
                        help='TP size for LLM')
    
    # Sampling parameters
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Temperature for sampling')
    parser.add_argument('--top-p', type=float, default=1.0,
                        help='Top-p for nucleus sampling')
    parser.add_argument('--top-k', type=int, default=-1,
                        help='Top-k for sampling')
    
    # Debug options
    parser.add_argument('--display-progress', action='store_true',
                        help='Display progress during generation')
    parser.add_argument('--overlap-tp-schedule', action='store_true',
                        help='Enable overlap TP schedule')
    
    # Output options
    parser.add_argument('--output-csv', type=str, default=None,
                        help='Path to output CSV file for results')
    args = parser.parse_args()

    print("=" * 60)
    print("SLDisaggregationSystem Speed Test")
    print("=" * 60)

    # Load config from path
    config_path = args.config_path
    if os.path.isdir(config_path):
        config_path = os.path.join(config_path, "config.yaml")
    with open(config_path, "r") as f:
        model_config = json.load(f)
    router_config = model_config.get("router", {})
    router_path = router_config.get("router_path")

    quick_sglang_kwargs = {"dtype": "bfloat16", "tp_size": args.slm_tp_size, "enable_return_hidden_states": True}
    reference_sglang_kwargs = {"dtype": "bfloat16", "tp_size": args.llm_tp_size}
    strategy_kwargs = {"model_path": router_path}
    
    # Determine switching strategy first
    switching_strategy = router_config.get("switching_strategy")
    if switching_strategy is None:
        switching_strategy = "neural"
    print(f"Using switching strategy: {switching_strategy}")

    # Threshold loading logic
    if switching_strategy == "neural":
        # Priority: config file's router.threshold > command line arg
        threshold = router_config.get("threshold")
        if threshold is None and args.threshold is not None:
            threshold = args.threshold
        
        if threshold is not None:
            strategy_kwargs["threshold"] = threshold
            print(f"Using neural threshold: {threshold}")
    else:
        # For non-neural strategies, use specific thresholds from config
        if "aleatoric_threshold" in router_config:
            strategy_kwargs["aleatoric_threshold"] = router_config["aleatoric_threshold"]
            print(f"Using aleatoric threshold from config: {router_config['aleatoric_threshold']}")
        
        if "entropy_threshold" in router_config:
            strategy_kwargs["entropy_threshold"] = router_config["entropy_threshold"]
            print(f"Using entropy threshold from config: {router_config['entropy_threshold']}")

    # Initialize SLDisaggregationSystem
    print("\nInitializing SLDisaggregationSystem...")
    generator = SLDisaggregationSystem(
        model_config=model_config,
        device="cuda",
        dtype=torch.bfloat16,
        switching_strategy=switching_strategy,
        strategy_kwargs=strategy_kwargs,
        is_record=True,
        quick_sglang_kwargs=quick_sglang_kwargs,
        reference_sglang_kwargs=reference_sglang_kwargs,
        overlap_tp_schedule=args.overlap_tp_schedule,
    )
    print("Generator initialized.\n")

    # Load prompt from input_text.txt
    input_file = os.path.join(os.path.dirname(__file__), "input_text.txt")
    if os.path.isfile(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            prompts = [ln.strip() for ln in f.readlines() if ln.strip()]
        prompts = prompts[:args.batch_size]
        print(f"Loaded prompt from {input_file}")
    else:
        prompt = "Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop."
        prompts = [prompt] * args.batch_size
        print(f"input_text.txt not found, using default prompts")
    
    # Prepare input ids
    batch_input_ids = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        prompt_text = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = generator.tokenizer.encode(prompt_text)
        batch_input_ids.append(input_ids)
    
    input_length = sum(len(input_ids) for input_ids in batch_input_ids) / len(batch_input_ids)
    print(f"Average Input Length: {input_length} tokens")
    print(f"Max Decode Length: {args.decode_length} tokens")
    print("=" * 60)

    # Warm-up parameters
    warmup_iterations = 5
    warmup_decode_length = 8192
    benchmark_iterations = 5

    # Prepare dummy input for warm-up
    dummy_prompt = "Hello, this is a warm-up test. Please generate some text."
    dummy_messages = [{"role": "user", "content": dummy_prompt}]
    dummy_prompt_text = generator.tokenizer.apply_chat_template(dummy_messages, tokenize=False, add_generation_prompt=True)
    dummy_input_ids = generator.tokenizer.encode(dummy_prompt_text)
    dummy_batch_input_ids = [dummy_input_ids]

    # Warm-up phase
    print(f"\n{'=' * 60}")
    print(f"WARM-UP PHASE: {warmup_iterations} iterations x {warmup_decode_length} tokens")
    print("=" * 60)
    
    for i in range(warmup_iterations):
        print(f"Warm-up iteration {i + 1}/{warmup_iterations}...", end=" ", flush=True)
        torch.cuda.synchronize()
        warmup_start = time.perf_counter()
        
        generator.generate(
            dummy_batch_input_ids,
            max_new_tokens=warmup_decode_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            display_progress=False,
        )
        
        torch.cuda.synchronize()
        warmup_end = time.perf_counter()
        print(f"done ({warmup_end - warmup_start:.2f}s)")
    
    print("Warm-up complete.\n")

    # Benchmark phase - run multiple times
    print(f"{'=' * 60}")
    print(f"BENCHMARK PHASE: {benchmark_iterations} iterations")
    print("=" * 60)
    
    benchmark_times = []
    benchmark_output_tokens = []
    benchmark_throughputs = []
    all_slm_counts = []
    all_llm_counts = []
    all_llm_ratios = []
    last_output_text = ""
    
    for i in range(benchmark_iterations):
        print(f"\nBenchmark iteration {i + 1}/{benchmark_iterations}...")
        torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        torch.cuda.cudart().cudaProfilerStart()
        result = generator.generate(
            batch_input_ids,
            max_new_tokens=args.decode_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            display_progress=args.display_progress,
        )
        
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        end_time = time.perf_counter()
        
        iter_time = end_time - start_time
        benchmark_times.append(iter_time)

        # Calculate output tokens and get generated text
        iter_output_tokens = 0
        slm_token_count = 0
        llm_token_count = 0
        llm_ratio = 0.0

        for obj in result:

            if isinstance(obj, dict):
                iter_output_tokens += len(obj.get("output_ids", []))
                output_ids = obj.get("output_ids", [])
                slm_token_count += obj.get("slm_token_count", 0)
                llm_token_count += obj.get("llm_token_count", 0)
            else:
                iter_output_tokens += len(obj.output_ids)
                output_ids = obj.output_ids
                slm_token_count += getattr(obj, "slm_token_count", 0)
                llm_token_count += getattr(obj, "llm_token_count", 0)

        total = slm_token_count + llm_token_count
        llm_ratio = llm_token_count / total if total > 0 else 0.0
        
        benchmark_output_tokens.append(iter_output_tokens)
        all_slm_counts.append(slm_token_count)
        all_llm_counts.append(llm_token_count)
        all_llm_ratios.append(llm_ratio)
        
        # Calculate throughput
        iter_throughput = iter_output_tokens / iter_time if iter_time > 0 else 0
        benchmark_throughputs.append(iter_throughput)
        
        # Decode output text (keep last one for CSV)
        last_output_text = generator.tokenizer.decode(output_ids, skip_special_tokens=True)
        
        print(f"  Time: {iter_time:.3f}s, Tokens: {iter_output_tokens}, Throughput: {iter_throughput:.2f} tokens/s, LLM Ratio: {llm_ratio:.2%}")

    # Calculate averages
    avg_time = sum(benchmark_times) / len(benchmark_times)
    avg_output_tokens = sum(benchmark_output_tokens) / len(benchmark_output_tokens)
    avg_throughput = sum(benchmark_throughputs) / len(benchmark_throughputs)
    avg_slm_count = sum(all_slm_counts) / len(all_slm_counts)
    avg_llm_count = sum(all_llm_counts) / len(all_llm_counts)
    avg_llm_ratio = sum(all_llm_ratios) / len(all_llm_ratios)

    # Print results
    print("\n" + "=" * 60)
    print("PERFORMANCE RESULTS - SLDisaggregationSystem")
    print(f"(Average of {benchmark_iterations} benchmark runs after {warmup_iterations} warm-up runs)")
    print("=" * 60)
    print(f"Average Total Time: {avg_time:.3f} seconds")
    print(f"")
    print(f"Input Tokens:  {input_length}")
    print(f"Average Output Tokens: {avg_output_tokens:.1f}")
    print(f"")
    print(f"Average Decode Throughput: {avg_throughput:.2f} tokens/s")
    print("=" * 60)
    print("LLM CALL STATISTICS (Average)")
    print("=" * 60)
    print(f"Average SLM (Quick Model) Tokens: {avg_slm_count:.1f}")
    print(f"Average LLM (Reference Model) Tokens: {avg_llm_count:.1f}")
    print(f"Average LLM Call Ratio: {avg_llm_ratio:.2%}")
    print(f"Average SLM Call Ratio: {1 - avg_llm_ratio:.2%}")
    print("=" * 60)
    print(f"Individual runs: {[f'{t:.2f}s' for t in benchmark_times]}")
    print(f"Throughputs: {[f'{t:.2f}' for t in benchmark_throughputs]} tokens/s")
    print("=" * 60)

    # Use average values for CSV output
    total_time = avg_time
    output_tokens = int(avg_output_tokens)
    decode_tokens_per_sec = avg_throughput
    llm_ratio = avg_llm_ratio
    output_text = last_output_text

    # Save results to CSV if specified
    if args.output_csv:
        results = {
            'timestamp': datetime.now().isoformat(),
            'system': 'SLDisaggregationSystem',
            'config_path': args.config_path,
            'threshold': threshold,
            'batch_size': args.batch_size,
            'input_tokens': input_length,
            'output_tokens': output_tokens,
            'total_time_s': round(total_time, 3),
            'decode_throughput_tps': round(decode_tokens_per_sec, 2),
            'quick_ratio': round(1 - llm_ratio, 4),
            'reference_ratio': round(llm_ratio, 4),
            'prompt': prompt,
            'generated_text': output_text,
        }
        save_results_to_csv(args.output_csv, results)

    print("\nTest complete.")


if __name__ == "__main__":
    if torch.cuda.is_available():
        pass
    else:
        print("WARNING: CUDA not available. SGLang dynamic mode will likely fail.")

    mp.set_start_method("spawn", force=True)
    main()
