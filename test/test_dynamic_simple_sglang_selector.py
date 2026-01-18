import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ.setdefault('MASTER_PORT', '29500')
os.environ["SGLANG_ENABLE_TORCH_COMPILE"] = "0"

import sys
import time
import argparse
import torch
import multiprocessing as mp
import warnings
import csv
from datetime import datetime

from r2r.models.dynamic_sglang_selector import DynamicSimpleSGLangSelector
import yaml

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
        'timestamp', 'system', 'config_path', 'threshold',
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
    parser = argparse.ArgumentParser(description="Test DynamicSimpleSGLangSelector - Speed Test")
    parser.add_argument('--config-path', type=str, 
                        default="config/local/DeepSeek-R1-Distill-Qwen-1.5B+DeepSeek-R1-Distill-Qwen-32B_local.yaml",
                        help='Path to config.yaml')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for neural routing (fallback if not in config)')
    parser.add_argument('--tp-size', type=int, default=None,
                        help='Tensor parallelism size (defaults to available GPU count)')
    
    # Test parameters
    parser.add_argument('--decode-length', type=int, default=8192,
                        help='Maximum number of tokens to decode/generate')
    
    # Sampling parameters
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Temperature for sampling')
    parser.add_argument('--top-p', type=float, default=1.0,
                        help='Top-p for nucleus sampling')
    parser.add_argument('--top-k', type=int, default=-1,
                        help='Top-k for sampling')
    
    # Debug options
    parser.add_argument('--print-tokens', action='store_true',
                        help='Print tokens during generation')
    parser.add_argument('--record-generation', action='store_true',
                        help='Record generation details')
    
    # Output options
    parser.add_argument('--output-csv', type=str, default=None,
                        help='Path to output CSV file for results')
    
    parser.add_argument('--record_generation', action='store_true',
                        help='Record generation details')
    args = parser.parse_args()

    print("=" * 60)
    print("DynamicSimpleSGLangSelector Speed Test")
    print("=" * 60)

    # Load config from path
    config_path = args.config_path
    if os.path.isdir(config_path):
        config_path = os.path.join(config_path, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)
    router_config = model_config.get("router", {})
    router_path = router_config.get("router_path")

    # Setup sglang kwargs
    sglang_kwargs = {"dtype": "bfloat16"}
    if args.tp_size is not None:
        sglang_kwargs["tp_size"] = args.tp_size
    else:
        sglang_kwargs["tp_size"] = torch.cuda.device_count()

    # Setup strategy kwargs
    strategy_kwargs = {"model_path": router_path}
    
    # Priority: config file's router.threshold > command line arg
    threshold = router_config.get("threshold")
    if threshold is None and args.threshold is not None:
        threshold = args.threshold
    
    if threshold is not None:
        strategy_kwargs["threshold"] = threshold
        print(f"Using neural threshold: {threshold}")

    # Initialize DynamicSimpleSGLangSelector
    print("\nInitializing DynamicSimpleSGLangSelector...")
    generator = DynamicSimpleSGLangSelector(
        model_config=model_config,
        device="cuda",
        dtype=torch.bfloat16,
        switching_strategy="neural",
        strategy_kwargs=strategy_kwargs,
        is_record=args.record_generation,
        sglang_kwargs=sglang_kwargs,
    )
    print("Generator initialized.\n")

    # Load prompt from input_text.txt
    input_file = os.path.join(os.path.dirname(__file__), "input_text.txt")
    if os.path.isfile(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        print(f"Loaded prompt from {input_file}")
    else:
        prompt = "Write a detailed essay about the history of artificial intelligence."
        print(f"input_text.txt not found, using default prompt")
    
    # Prepare input ids (batch_size = 1)
    messages = [{"role": "user", "content": prompt}]
    prompt_text = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = generator.tokenizer.encode(prompt_text)
    batch_input_ids = [input_ids]
    
    input_length = len(input_ids)
    print(f"Input Length: {input_length} tokens")
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
            record_generation=False,
            print_tokens=False,
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
    all_quick_ratios = []
    all_reference_ratios = []
    
    for i in range(benchmark_iterations):
        print(f"\nBenchmark iteration {i + 1}/{benchmark_iterations}...")
        torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        generated_texts, recorders = generator.generate(
            batch_input_ids,
            max_new_tokens=args.decode_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            record_generation=args.record_generation,
            print_tokens=args.print_tokens,
        )
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        iter_time = end_time - start_time
        benchmark_times.append(iter_time)
        
        # Calculate output tokens
        output_text = generated_texts[0]
        iter_output_tokens = len(generator.tokenizer.encode(output_text))
        benchmark_output_tokens.append(iter_output_tokens)
        
        # Calculate throughput
        iter_throughput = iter_output_tokens / iter_time if iter_time > 0 else 0
        benchmark_throughputs.append(iter_throughput)
        
        # Calculate quick/reference ratio
        quick_ratio = 0.0
        reference_ratio = 0.0
        if args.record_generation and recorders:
            recorder = recorders[0]
            if hasattr(recorder, 'records') and recorder.records:
                quick_count = sum(1 for r in recorder.records if r.source_model == "quick")
                reference_count = sum(1 for r in recorder.records if r.source_model == "reference")
                total = len(recorder.records)
                quick_ratio = quick_count / total if total > 0 else 0
                reference_ratio = reference_count / total if total > 0 else 0
        
        all_quick_ratios.append(quick_ratio)
        all_reference_ratios.append(reference_ratio)
        
        print(f"  Time: {iter_time:.3f}s, Tokens: {iter_output_tokens}, Throughput: {iter_throughput:.2f} tokens/s")

    # Calculate averages
    avg_time = sum(benchmark_times) / len(benchmark_times)
    avg_output_tokens = sum(benchmark_output_tokens) / len(benchmark_output_tokens)
    avg_throughput = sum(benchmark_throughputs) / len(benchmark_throughputs)
    avg_quick_ratio = sum(all_quick_ratios) / len(all_quick_ratios) if all_quick_ratios else 0
    avg_reference_ratio = sum(all_reference_ratios) / len(all_reference_ratios) if all_reference_ratios else 0

    # Print results
    print("\n" + "=" * 60)
    print("PERFORMANCE RESULTS - DynamicSimpleSGLangSelector")
    print(f"(Average of {benchmark_iterations} benchmark runs after {warmup_iterations} warm-up runs)")
    print("=" * 60)
    print(f"Average Total Time: {avg_time:.3f} seconds")
    print(f"")
    print(f"Input Tokens:  {input_length}")
    print(f"Average Output Tokens: {avg_output_tokens:.1f}")
    print(f"")
    print(f"Average Decode Throughput: {avg_throughput:.2f} tokens/s")
    print("=" * 60)
    print(f"Individual runs: {[f'{t:.2f}s' for t in benchmark_times]}")
    print(f"Throughputs: {[f'{t:.2f}' for t in benchmark_throughputs]} tokens/s")
    print("=" * 60)

    # Print recorder statistics if available
    if args.record_generation:
        print(f"\n=== Generation Records (Average) ===")
        print(f"Quick: {avg_quick_ratio*100:.1f}%")
        print(f"Reference: {avg_reference_ratio*100:.1f}%")
    
    # Use last iteration values for CSV output
    total_time = avg_time
    output_tokens = int(avg_output_tokens)
    decode_tokens_per_sec = avg_throughput
    quick_ratio = avg_quick_ratio
    reference_ratio = avg_reference_ratio

    # Save results to CSV if specified
    if args.output_csv:
        results = {
            'timestamp': datetime.now().isoformat(),
            'system': 'DynamicSimpleSGLangSelector',
            'config_path': args.config_path,
            'threshold': threshold,
            'input_tokens': input_length,
            'output_tokens': output_tokens,
            'total_time_s': round(total_time, 3),
            'decode_throughput_tps': round(decode_tokens_per_sec, 2),
            'quick_ratio': round(quick_ratio, 4) if args.record_generation else None,
            'reference_ratio': round(reference_ratio, 4) if args.record_generation else None,
            'prompt': prompt,
            'generated_text': output_text,
        }
        save_results_to_csv(args.output_csv, results)

    # Shutdown generator
    generator.shutdown()
    print("\nGenerator shutdown complete.")


if __name__ == "__main__":
    if torch.cuda.is_available():
        pass
    else:
        print("WARNING: CUDA not available. DynamicSimpleSGLangSelector will likely fail.")

    mp.set_start_method("spawn", force=True)
    main()
