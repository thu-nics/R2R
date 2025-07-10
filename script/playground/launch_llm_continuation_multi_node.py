"""
Multi-node distributed LLM continuation launcher using Ray.

This script distributes the LLM continuation task across multiple nodes using Ray,
with each job using 2 GPUs (tp_size=2, dp_size=1).

Usage:
    python launch_llm_continuation_multi_node.py --input_path <input_csv> --output_path <output_dir> --num_jobs <num_jobs>
"""

import os
import sys
import argparse
import pandas as pd
import ray
from pathlib import Path
import logging
import time
import json
from typing import List, Dict, Any, Tuple
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1)
def find_available_gpu_pairs(min_free_mem_gb: int) -> List[Tuple[int, int]]:
    """
    Finds pairs of available GPUs on the current node that meet the memory requirement.
    """
    try:
        import pynvml
    except ImportError:
        logger.warning(
            "pynvml not found. Cannot check GPU memory for smart scheduling. "
            "Please run 'pip install pynvml' to enable this feature. "
            "Falling back to default Ray scheduling if possible, but custom logic will fail."
        )
        return []

    gpu_pairs = []
    try:
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        available_gpus = []
        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_mem_gb = mem_info.free / (1024**3)
            used_mem_mb = mem_info.used / (1024**2)

            # A GPU is considered available if it has enough free memory and is not heavily used.
            # A small amount of used memory is acceptable (e.g., for display servers).
            if free_mem_gb >= min_free_mem_gb and used_mem_mb < 1000:  # Less than 1GB used
                available_gpus.append(i)
        
        # Create pairs from the list of available GPUs
        for i in range(0, len(available_gpus) - 1, 2):
            gpu_pairs.append((available_gpus[i], available_gpus[i+1]))
            
        if gpu_pairs:
            logger.info(f"Node has {len(gpu_pairs)} available GPU pair(s) with >{min_free_mem_gb}GB free memory each: {gpu_pairs}")
            
        pynvml.nvmlShutdown()
    except pynvml.NVMLError as e:
        logger.warning(f"Could not use NVML on this node: {e}. Assuming no GPUs available for custom scheduling here.")
    
    return gpu_pairs


@ray.remote(num_gpus=3)
class LLMContinuationWorker:
    """Ray actor for running LLM continuation on a subset of data."""
    
    def __init__(self, gpu_ids: Tuple[int, int]):
        """Initialize the worker with specific GPU IDs."""
        self.gpu_ids = gpu_ids
        logger.info(f"Worker actor initialized for GPU pair {self.gpu_ids}")
    
    def run_continuation(self, 
                        input_path: str,
                        output_path: str, 
                        low: int, 
                        high: int,
                        **kwargs) -> Dict[str, Any]:
        """
        Run LLM continuation for a specific data range.
        
        Args:
            input_path: Path to input CSV file
            output_path: Path to output directory  
            low: Lower bound of data sample ID range (inclusive)
            high: Upper bound of data sample ID range (exclusive)
            **kwargs: Additional arguments to pass to step_2_llm_continuation.py
            
        Returns:
            Dictionary with execution results and metadata
        """
        import subprocess
        import sys
        from pathlib import Path
        
        # Get the directory of the current script
        script_dir = Path(__file__).parent
        continuation_script = script_dir / "step_2_llm_continuation.py"
        
        # Build command arguments
        cmd = [
            sys.executable, str(continuation_script),
            "--input_path", input_path,
            "--output_path", output_path,
            "--low", str(low),
            "--high", str(high),
            "--tp_size", "2",  # Fixed tp_size as requested
            "--dp_size", "1",  # Fixed dp_size as requested
        ]
        
        # Add additional arguments
        for key, value in kwargs.items():
            if key in ['input_path', 'output_path', 'low', 'high', 'tp_size', 'dp_size']:
                continue  # Skip already handled arguments
            cmd.extend([f"--{key}", str(value)])
        
        # Set CUDA_VISIBLE_DEVICES for the subprocess to use the assigned GPUs
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = f"{self.gpu_ids[0]},{self.gpu_ids[1]}"
        
        start_time = time.time()
        
        try:
            logger.info(f"Worker on GPUs {self.gpu_ids} starting job for range [{low}, {high}) with command: {' '.join(cmd)}")
            
            # Run the subprocess with the specific GPU environment
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            logger.info(f"Worker on GPUs {self.gpu_ids} completed job for range [{low}, {high}) in {execution_time:.2f} seconds")
            
            return {
                'success': True,
                'gpus': self.gpu_ids,
                'low': low,
                'high': high,
                'execution_time': execution_time,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
        except subprocess.CalledProcessError as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            logger.error(f"Worker on GPUs {self.gpu_ids} failed job for range [{low}, {high}) after {execution_time:.2f} seconds")
            logger.error(f"Error: {e}")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
            
            return {
                'success': False,
                'gpus': self.gpu_ids,
                'low': low,
                'high': high,
                'execution_time': execution_time,
                'stdout': e.stdout,
                'stderr': e.stderr,
                'return_code': e.returncode,
                'error': str(e)
            }


def split_data_ranges(total_samples: int, num_jobs: int) -> List[tuple]:
    """
    Split the data range into num_jobs chunks.
    
    Args:
        total_samples: Total number of data samples
        num_jobs: Number of jobs to split into
        
    Returns:
        List of (low, high) tuples representing data ranges
    """
    if num_jobs <= 0:
        raise ValueError("num_jobs must be positive")
    
    if total_samples <= 0:
        raise ValueError("total_samples must be positive")
    
    chunk_size = total_samples // num_jobs
    remainder = total_samples % num_jobs
    
    ranges = []
    current_low = 0
    
    for i in range(num_jobs):
        # Add 1 to chunk_size for the first 'remainder' jobs to distribute remainder evenly
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        current_high = current_low + current_chunk_size
        
        if current_low < total_samples:  # Only add if there's data to process
            ranges.append((current_low, min(current_high, total_samples)))
            current_low = current_high
        
    return ranges


def merge_results(output_path: Path, ranges: List[tuple], final_filename: str) -> None:
    """
    Merge results from all workers into a single file.
    
    Args:
        output_path: Output directory path
        ranges: List of (low, high) tuples that were processed
        final_filename: Name of the final merged file
    """
    logger.info("Starting to merge results from all workers...")
    
    all_dataframes = []
    
    for low, high in ranges:
        # Look for the individual result file
        individual_file = output_path / f"generation_results_data_{low}_to_{high}_real.csv"
        
        if individual_file.exists():
            logger.info(f"Loading results from {individual_file}")
            df = pd.read_csv(individual_file)
            all_dataframes.append(df)
        else:
            logger.warning(f"Result file not found: {individual_file}")
    
    if all_dataframes:
        # Concatenate all dataframes
        merged_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Save merged results
        final_path = output_path / final_filename
        merged_df.to_csv(final_path, index=False)
        
        logger.info(f"Merged {len(all_dataframes)} result files into {final_path}")
        logger.info(f"Total rows in merged file: {len(merged_df)}")
        
        # Optionally remove individual files to save space
        # for low, high in ranges:
        #     individual_file = output_path / f"generation_results_data_{low}_to_{high}_real.csv"
        #     if individual_file.exists():
        #         individual_file.unlink()
        #         logger.info(f"Removed individual file: {individual_file}")
    else:
        logger.error("No result files found to merge!")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Distributed LLM continuation using Ray')
    
    # Required arguments
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input CSV file from step 1')
    parser.add_argument('--output_path', type=str, default='output/playground/continuation_distributed',
                        help='Path to store output files')
    parser.add_argument('--num_jobs', type=int, required=True,
                        help='Number of parallel jobs to run')
    
    # Ray configuration
    parser.add_argument('--ray_address', type=str, default=None,
                        help='Ray cluster address (e.g., "ray://head_node_ip:10001"). If None, will start local cluster.')
    parser.add_argument('--ray_runtime_env', type=str, default=None,
                        help='JSON string for Ray runtime environment configuration')
    
    # Smart scheduling parameters
    parser.add_argument('--min_gpu_mem_gb', type=int, default=40,
                        help='Minimum free GPU memory in GB required for a GPU to be considered available for a worker.')

    # Model parameters (passed to each worker)
    parser.add_argument('--max_tokens', type=int, default=32768,
                        help='Maximum number of tokens per data sample')
    parser.add_argument('--max_new_tokens', type=int, default=8192,
                        help='Maximum number of new tokens to generate')
    parser.add_argument('--gen_mem_fraction', type=float, default=0.9,
                        help='Fraction of GPU memory to allocate for generation')
    parser.add_argument('--verify_mem_fraction', type=float, default=0.3,
                        help='Fraction of GPU memory to allocate for judging')
    parser.add_argument('--num_continuation', type=int, default=1,
                        help='Number of continuations to generate')
    parser.add_argument('--previous_context', type=int, default=0,
                        help='Number of previous sentences to include in the context')
    parser.add_argument('--common_previous_context', type=int, default=-1,
                        help='Number of previous sentences to include in the common context')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of small model outputs to generate for each mismatch point')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=1.0,
                        help='Top-p probability threshold for nucleus sampling')
    parser.add_argument('--top_k', type=int, default=-1,
                        help='Top-k sampling parameter for generation')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of mismatches to batch together for generation')
    parser.add_argument('--skip_stress_divergent_token', action='store_true', default=False,
                        help='Whether to skip stressing the divergent token')
    parser.add_argument('--is_print', action='store_true', default=False,
                        help='Whether to print the results')
    
    return parser.parse_args()


def main():
    """Main function to orchestrate distributed LLM continuation."""
    args = parse_args()
    
    # Create output directory
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = args.__dict__.copy()
    with open(output_path / 'distributed_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Load input data to determine total number of samples
    logger.info(f"Loading input data from {args.input_path}")
    input_data = pd.read_csv(args.input_path)
    
    # Get unique data_ids to determine the range
    unique_data_ids = sorted(input_data['data_id'].unique())
    total_samples = len(unique_data_ids)
    min_data_id = min(unique_data_ids)
    max_data_id = max(unique_data_ids)
    
    logger.info(f"Found {total_samples} unique data samples (IDs: {min_data_id} to {max_data_id})")
    
    # Split data into ranges
    ranges = split_data_ranges(total_samples, args.num_jobs)
    logger.info(f"Split data into {len(ranges)} jobs:")
    for i, (low, high) in enumerate(ranges):
        # Convert from index-based to actual data_id range
        actual_low = unique_data_ids[low] if low < len(unique_data_ids) else unique_data_ids[-1]
        actual_high = unique_data_ids[high-1] + 1 if high <= len(unique_data_ids) else unique_data_ids[-1] + 1
        logger.info(f"  Job {i+1}: data_id range [{actual_low}, {actual_high}) ({high-low} samples)")
    
    # Initialize Ray
    runtime_env = None
    if args.ray_runtime_env:
        runtime_env = json.loads(args.ray_runtime_env)
    
    if args.ray_address:
        logger.info(f"Connecting to Ray cluster at {args.ray_address}")
        ray.init(address=args.ray_address, runtime_env=runtime_env)
    else:
        logger.info("Starting local Ray cluster")
        ray.init(runtime_env=runtime_env)
    
    logger.info(f"Ray cluster initialized with {ray.cluster_resources()} resources")
    
    try:
        # --- Smart GPU Scheduling Logic ---
        logger.info("Scanning for available GPU pairs across the cluster...")
        # Get nodes that are alive and have GPUs
        nodes = [node for node in ray.nodes() if node.get("Alive") and node.get("Resources", {}).get("GPU", 0) > 0]
        if not nodes:
            logger.error("No nodes with GPU resources found in the Ray cluster. Exiting.")
            ray.shutdown()
            return

        logger.info(f"Found {len(nodes)} nodes with GPUs. Checking for available pairs with >{args.min_gpu_mem_gb}GB free memory.")

        # Find available GPU pairs on each node by scheduling a check on each specific node
        check_futures = []
        for node in nodes:
            node_resource = f"node:{node['NodeAddress']}"
            check_futures.append(
                find_available_gpu_pairs.options(resources={node_resource: 0.01}).remote(args.min_gpu_mem_gb)
            )
        results_per_node = ray.get(check_futures)

        all_gpu_pairs = []
        for i, node_gpu_pairs in enumerate(results_per_node):
            node_address = nodes[i]['NodeAddress']
            for pair in node_gpu_pairs:
                all_gpu_pairs.append({'node_address': node_address, 'gpus': pair})

        if not all_gpu_pairs:
            logger.error(f"No available GPU pairs found with at least {args.min_gpu_mem_gb}GB free memory. Exiting.")
            ray.shutdown()
            return

        logger.info(f"Found a total of {len(all_gpu_pairs)} available GPU pairs across the cluster.")

        # Determine the number of jobs based on available resources and user request
        num_jobs = min(args.num_jobs, len(all_gpu_pairs))
        if num_jobs < args.num_jobs:
            logger.warning(f"User requested {args.num_jobs} jobs, but only {len(all_gpu_pairs)} GPU pairs are available. "
                           f"Running {num_jobs} jobs instead.")
        if num_jobs == 0:
            logger.error("Number of jobs is 0. Either --num_jobs was 0 or no suitable GPU pairs were found. Exiting.")
            ray.shutdown()
            return

        # Re-split ranges based on the actual number of jobs we can run
        ranges = split_data_ranges(total_samples, num_jobs)

        # Create workers on the specific nodes and GPUs
        workers = []
        for i in range(num_jobs):
            gpu_spec = all_gpu_pairs[i]
            node_address = gpu_spec['node_address']
            gpu_ids = gpu_spec['gpus']
            
            node_resource = f"node:{node_address}"
            worker = LLMContinuationWorker.options(resources={node_resource: 0.01}).remote(gpu_ids=gpu_ids)
            workers.append(worker)
            
        logger.info(f"Created {len(workers)} workers on specific GPU pairs.")
        
        # Prepare arguments for workers (exclude Ray-specific args)
        worker_kwargs = {k: v for k, v in args.__dict__.items() 
                        if k not in ['input_path', 'output_path', 'num_jobs', 'ray_address', 'ray_runtime_env', 'min_gpu_mem_gb']}
        
        # Submit jobs
        futures = []
        for i, (worker, (low, high)) in enumerate(zip(workers, ranges)):
            # Convert index-based range to actual data_id range
            actual_low = unique_data_ids[low] if low < len(unique_data_ids) else unique_data_ids[-1]
            actual_high = unique_data_ids[high-1] + 1 if high <= len(unique_data_ids) else unique_data_ids[-1] + 1
            
            future = worker.run_continuation.remote(
                input_path=args.input_path,
                output_path=args.output_path,
                low=actual_low,
                high=actual_high,
                **worker_kwargs
            )
            futures.append(future)
            logger.info(f"Submitted job {i+1}/{len(ranges)} for data_id range [{actual_low}, {actual_high}) to worker on node {all_gpu_pairs[i]['node_address']} with GPUs {all_gpu_pairs[i]['gpus']}")
        
        # Wait for all jobs to complete
        logger.info("Waiting for all jobs to complete...")
        start_time = time.time()
        
        results = ray.get(futures)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Process results
        successful_jobs = [r for r in results if r['success']]
        failed_jobs = [r for r in results if not r['success']]
        
        logger.info(f"All jobs completed in {total_time:.2f} seconds")
        logger.info(f"Successful jobs: {len(successful_jobs)}/{len(results)}")
        
        if failed_jobs:
            logger.error(f"Failed jobs: {len(failed_jobs)}")
            for job in failed_jobs:
                logger.error(f"  Job [{job['low']}, {job['high']}): {job.get('error', 'Unknown error')}")
        
        # Merge results if we have successful jobs
        if successful_jobs:
            # Convert back to index-based ranges for merging
            successful_ranges = []
            for job in successful_jobs:
                # Find the index range corresponding to this data_id range
                low_idx = unique_data_ids.index(job['low'])
                # Find the last data_id that's < job['high']
                high_idx = len(unique_data_ids)
                for i, data_id in enumerate(unique_data_ids):
                    if data_id >= job['high']:
                        high_idx = i
                        break
                successful_ranges.append((job['low'], job['high']))
            
            merge_results(output_path, successful_ranges, "generation_results_data_all_real_merged.csv")
            
            # Save execution summary
            summary = {
                'total_jobs': len(results),
                'successful_jobs': len(successful_jobs),
                'failed_jobs': len(failed_jobs),
                'total_execution_time': total_time,
                'average_job_time': sum(r['execution_time'] for r in successful_jobs) / len(successful_jobs) if successful_jobs else 0,
                'job_results': results
            }
            
            with open(output_path / 'execution_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Execution summary saved to {output_path / 'execution_summary.json'}")
        else:
            logger.error("No successful jobs to merge!")
    
    finally:
        # Shutdown Ray
        ray.shutdown()
        logger.info("Ray cluster shutdown complete")


if __name__ == "__main__":
    main()
