#!/usr/bin/env python3
"""
Intelligent automated task scheduler for LLM continuation.

This script provides a fully automated solution that:
1. Automatically detects available GPUs
2. Dynamically assigns tasks to available GPUs
3. Skips already completed tasks by checking output files
4. Monitors task completion and automatically merges results
5. Provides real-time progress tracking

Usage:
    python auto_scheduler.py --input_path <input_csv> --output_path <output_dir> --num_workers <num_workers>
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
import logging
import json
import time
import subprocess
import threading
import multiprocessing as mp
from typing import List, Dict, Tuple, Optional, Set, Any
from datetime import datetime, timedelta
import signal
import psutil
import numpy as np

# Try to import pynvml for GPU detection
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("Warning: pynvml not available. GPU memory checking disabled.")
    print("Install with: pip install pynvml")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def make_json_serializable(obj: Any) -> Any:
    """Convert object to JSON serializable format, handling numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif hasattr(obj, 'item'):  # handles numpy scalars
        return obj.item()
    else:
        return obj


class GPUDetector:
    """GPU detection and allocation management."""
    
    def __init__(self, min_free_memory_gb: float = 10.0, gpus_per_worker: int = 2):
        self.min_free_memory_gb = min_free_memory_gb
        self.gpus_per_worker = gpus_per_worker
        self.allocated_gpus: Set[int] = set()
        
    def detect_available_gpus(self) -> List[int]:
        """Detect available GPUs with sufficient free memory."""
        available_gpus = []
        
        if not PYNVML_AVAILABLE:
            # Fallback: assume GPUs 0-7 are available
            logger.warning("PYNVML not available, assuming GPUs 0-7 are available")
            import torch
            if torch.cuda.is_available():
                return list(range(torch.cuda.device_count()))
            return []
        
        try:
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(gpu_count):
                if i in self.allocated_gpus:
                    continue
                    
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                free_mem_gb = mem_info.free / (1024**3)
                used_mem_mb = mem_info.used / (1024**2)
                
                if free_mem_gb >= self.min_free_memory_gb and used_mem_mb < 1000:
                    available_gpus.append(i)
                    
            pynvml.nvmlShutdown()
            
        except Exception as e:
            logger.error(f"Error detecting GPUs: {e}")
            
        return available_gpus
    
    def get_gpu_pairs(self, num_workers: int) -> List[List[int]]:
        """Get GPU pairs for workers."""
        available_gpus = self.detect_available_gpus()
        logger.info(f"Available GPUs: {available_gpus}")
        
        gpu_pairs = []
        available_gpus = [gpu for gpu in available_gpus if gpu not in self.allocated_gpus]
        
        for i in range(0, len(available_gpus), self.gpus_per_worker):
            if len(gpu_pairs) >= num_workers:
                break
            if i + self.gpus_per_worker - 1 < len(available_gpus):
                pair = available_gpus[i:i + self.gpus_per_worker]
                gpu_pairs.append(pair)
                self.allocated_gpus.update(pair)
        
        logger.info(f"Allocated GPU pairs: {gpu_pairs}")
        return gpu_pairs
    
    def release_gpus(self, gpu_list: List[int]):
        """Release GPUs back to available pool."""
        for gpu in gpu_list:
            self.allocated_gpus.discard(gpu)
        logger.info(f"Released GPUs: {gpu_list}")

class TaskTracker:
    """Track task completion and progress."""
    
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.started_tasks: Set[Tuple[int, int]] = set()  # Tasks that have args files
        self.running_tasks: Dict[Tuple[int, int], threading.Thread] = {}
        self.lock = threading.Lock()
        
    def get_started_tasks(self) -> Set[Tuple[int, int]]:
        """Check output directory for tasks that have args files (should be skipped)."""
        started = set()
        
        if not self.output_path.exists():
            return started
            
        # Look for worker args files with pattern: worker_args_{low}_{high}.json
        for file_path in self.output_path.glob("worker_args_*_*.json"):
            try:
                # Extract range from filename
                filename = file_path.name
                parts = filename.replace("worker_args_", "").replace(".json", "").split("_")
                if len(parts) == 2:
                    low, high = int(parts[0]), int(parts[1])
                    
                    # If args file exists and has content, task was started
                    if file_path.stat().st_size > 0:
                        started.add((low, high))
                        logger.info(f"Found existing task args file: [{low}, {high})")
                            
            except ValueError:
                continue
                
        self.started_tasks = started
        logger.info(f"Found {len(started)} tasks with existing args files (will be skipped)")
        return started
    
    def is_task_started(self, low: int, high: int) -> bool:
        """Check if a specific task has been started (has args file)."""
        return (low, high) in self.started_tasks
    
    def mark_task_running(self, low: int, high: int, thread: threading.Thread):
        """Mark task as currently running."""
        with self.lock:
            logger.info(f"Marking task as running: [{low}, {high}]")
            self.running_tasks[(low, high)] = thread
    
    def mark_task_completed(self, low: int, high: int):
        """Mark task as completed."""
        with self.lock:
            logger.info(f"Marking task as completed: [{low}, {high}]")
            self.started_tasks.add((low, high))
            if (low, high) in self.running_tasks:
                del self.running_tasks[(low, high)]
    
    def get_running_tasks(self) -> Dict[Tuple[int, int], threading.Thread]:
        """Get currently running tasks."""
        with self.lock:
            return self.running_tasks.copy()


class WorkerManager:
    """Manage worker processes for LLM continuation tasks."""
    
    def __init__(self, args: argparse.Namespace, gpu_detector: GPUDetector, task_tracker: TaskTracker):
        self.args = args
        self.gpu_detector = gpu_detector
        self.task_tracker = task_tracker
        self.active_processes: List[subprocess.Popen] = []
        self.process_metadata: Dict[subprocess.Popen, Dict] = {}  # Track process metadata
        self.output_lock = threading.Lock()  # For thread-safe output
        
    def is_managed_process(self, process: subprocess.Popen) -> bool:
        """Verify if a process is managed by this worker manager."""
        return process in self.process_metadata and \
               self.process_metadata[process].get('scheduler_id') == self.args.scheduler_id
        
    def create_worker_command(self, low: int, high: int, gpu_ids: List[int]) -> List[str]:
        """Create command for worker process."""
        script_dir = Path(__file__).parent
        worker_script = script_dir / "step_2_llm_continuation.py"
        
        cmd = [
            sys.executable, str(worker_script),
            "--input_path", str(self.args.input_path),
            "--output_path", str(self.args.output_path),
            "--low", str(low),
            "--high", str(high),
            "--tp_size", str(self.args.tp_size),
            "--dp_size", str(self.args.dp_size),
            "--resume",
        ]
        
        # Add model parameters
        model_params = [
            'max_tokens', 'max_new_tokens', 'gen_mem_fraction', 'verify_mem_fraction',
            'num_continuation', 'previous_context', 'common_previous_context',
            'num_samples', 'temperature', 'top_p', 'top_k', 'batch_size'
        ]
        
        for param in model_params:
            if hasattr(self.args, param):
                value = getattr(self.args, param)
                cmd.extend([f"--{param}", str(value)])
        
        # Add boolean flags
        boolean_flags = ['skip_stress_divergent_token', 'is_print']
        for flag in boolean_flags:
            if hasattr(self.args, flag) and getattr(self.args, flag):
                cmd.append(f"--{flag}")
        
        return cmd
    
    def save_worker_args(self, low: int, high: int, gpu_ids: List[int], cmd: List[str]) -> Path:
        """Save worker arguments to a file for tracking purposes."""
        args_data = {
            'low': low,
            'high': high,
            'gpu_ids': gpu_ids,
            'command': cmd,
            'scheduler_id': self.args.scheduler_id,
            'start_time': datetime.now().isoformat()
        }
        
        # Make sure all data is JSON serializable
        serializable_data = make_json_serializable(args_data)
        
        # Create args filename
        args_filename = f"worker_args_{low}_{high}.json"
        args_path = Path(self.args.output_path) / args_filename
        
        # Save args file
        with open(args_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
            
        logger.info(f"Saved worker args to: {args_path}")
        return args_path

    def run_worker(self, low: int, high: int, gpu_ids: List[int]) -> bool:
        """Run a worker for the specified data range."""
        cmd = self.create_worker_command(low, high, gpu_ids)
        
        # Save worker args file before starting (marks task as started)
        args_path = self.save_worker_args(low, high, gpu_ids, cmd)
        
        # Set GPU environment and scheduler tracking
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        env["R2R_SCHEDULER_ID"] = self.args.scheduler_id
        env["R2R_WORKER_RANGE"] = f"{low}_{high}"
        
        # Print worker startup information
        logger.info(f"Starting worker for range [{low}, {high}) on GPUs {gpu_ids}")
        
        if self.args.show_worker_output:
            logger.info("=" * 80)
            logger.info(f"Worker Command for range [{low}, {high}):")
            logger.info(f"CUDA_VISIBLE_DEVICES={','.join(map(str, gpu_ids))} \\")
            cmd_str = " \\\n    ".join(cmd)
            logger.info(f"{cmd_str}")
            logger.info("=" * 80)
        
        try:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            self.active_processes.append(process)
            
            # Track process metadata
            self.process_metadata[process] = {
                'range': (low, high),
                'gpu_ids': gpu_ids,
                'scheduler_id': self.args.scheduler_id,
                'start_time': time.time(),
                'command': cmd
            }
            
            # Real-time output monitoring
            if self.args.show_worker_output:
                with self.output_lock:
                    logger.info(f"Worker [{low}, {high}) output:")
                    logger.info("-" * 60)
                
                # Read output line by line in real-time
                for line in iter(process.stdout.readline, ''):
                    if line:
                        # Add worker prefix to each line with thread-safe output
                        prefixed_line = f"[Worker {low}-{high}] {line.rstrip()}"
                        with self.output_lock:
                            print(prefixed_line, flush=True)  # Use print for immediate output
                        
                # Wait for process to complete
                process.wait()
                
                with self.output_lock:
                    logger.info("-" * 60)
            else:
                # Silent mode - just wait for completion
                process.wait()
            
            if process.returncode == 0:
                logger.info(f"Worker [{low}, {high}) completed successfully")
                self.task_tracker.mark_task_completed(low, high)
                return True
            else:
                logger.error(f"Worker [{low}, {high}) failed with return code {process.returncode}")
                return False
                
        except Exception as e:
            logger.error(f"Error running worker for range [{low}, {high}): {e}")
            return False
        finally:
            # Release GPUs
            self.gpu_detector.release_gpus(gpu_ids)
            if process in self.active_processes:
                self.active_processes.remove(process)
            if process in self.process_metadata:
                del self.process_metadata[process]
    
    def cleanup(self):

        if not self.active_processes:
            logger.info("No worker processes to clean up")
            return
            
        logger.info(f"Cleaning up {len(self.active_processes)} managed worker processes...")
        
        processes_to_cleanup = list(self.active_processes)  # Create a copy to avoid modification during iteration
        
        for i, process in enumerate(processes_to_cleanup):
            try:
                # Verify this is a process we manage
                if not self.is_managed_process(process):
                    logger.warning(f"Skipping process {process.pid} - not managed by this scheduler")
                    continue
                
                # Get metadata for this process
                metadata = self.process_metadata.get(process, {})
                range_info = metadata.get('range', 'unknown')
                gpu_info = metadata.get('gpu_ids', 'unknown')
                
                if process.poll() is None:  # Process is still running
                    logger.info(f"Terminating worker process {i+1}/{len(processes_to_cleanup)} "
                               f"(PID: {process.pid}, range: {range_info}, GPUs: {gpu_info})")
                    process.terminate()
                    
                    # Give process time to terminate gracefully
                    try:
                        process.wait(timeout=10)
                        logger.info(f"Worker process {process.pid} terminated gracefully")
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Worker process {process.pid} did not terminate gracefully, force killing...")
                        process.kill()
                        process.wait()
                        logger.info(f"Worker process {process.pid} force killed")
                else:
                    logger.info(f"Worker process {process.pid} (range: {range_info}) already finished")
                    
                # Release GPUs for this process
                if 'gpu_ids' in metadata:
                    self.gpu_detector.release_gpus(metadata['gpu_ids'])
                    
            except Exception as e:
                logger.error(f"Error cleaning up worker process {process.pid}: {e}")
                
        # Clear tracking structures
        self.active_processes.clear()
        self.process_metadata.clear()
        logger.info("Worker process cleanup completed!")


class AutoScheduler:
    """Main automatic scheduler class."""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output_path = Path(args.output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Track scheduler instance
        self.scheduler_id = f"auto_scheduler_{os.getpid()}_{int(time.time())}"
        
        # Initialize components
        self.gpu_detector = GPUDetector(
            min_free_memory_gb=args.min_gpu_memory, 
            gpus_per_worker=args.gpus_per_worker
        )
        self.task_tracker = TaskTracker(self.output_path)
        
        # Add scheduler_id to args for worker tracking
        args.scheduler_id = self.scheduler_id
        self.worker_manager = WorkerManager(args, self.gpu_detector, self.task_tracker)
        
        # Save configuration
        self.save_config()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def save_config(self):
        """Save scheduler configuration."""
        config = {
            'args': self.args.__dict__,
            'timestamp': datetime.now().isoformat(),
            'scheduler_id': self.scheduler_id,
            'process_id': os.getpid()
        }
        
        # Make sure config is JSON serializable
        serializable_config = make_json_serializable(config)
        
        with open(self.output_path / 'scheduler_config.json', 'w') as f:
            json.dump(serializable_config, f, indent=2)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.cleanup()
        sys.exit(0)
    
    def split_data_ranges(self, total_samples: int, num_workers: int) -> List[Tuple[int, int]]:
        """Split data into ranges for workers."""
        if num_workers <= 0:
            raise ValueError("num_workers must be positive")
        
        chunk_size = total_samples // num_workers
        remainder = total_samples % num_workers
        
        ranges = []
        current_low = 0
        
        for i in range(num_workers):
            current_chunk_size = chunk_size + (1 if i < remainder else 0)
            current_high = current_low + current_chunk_size
            
            if current_low < total_samples:
                ranges.append((current_low, min(current_high, total_samples)))
                current_low = current_high
        
        return ranges
    
    def get_pending_tasks(self, all_ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Get tasks that need to be run (no args file exists)."""
        # First scan for existing args files
        self.task_tracker.get_started_tasks()
        
        pending_tasks = []
        skipped_count = 0
        
        for low, high in all_ranges:
            if self.task_tracker.is_task_started(low, high):
                # Task has args file, skip it
                skipped_count += 1
                logger.info(f"Skipping task [{low}, {high}) - args file exists")
            else:
                # Task has no args file, need to run it
                pending_tasks.append((low, high))
        
        logger.info(f"Task status: {skipped_count} skipped (args file exists), {len(pending_tasks)} pending")
        return pending_tasks
    
    def run_tasks(self, pending_tasks: List[Tuple[int, int]]):
        """Run pending tasks using available GPUs."""
        if not pending_tasks:
            logger.info("No pending tasks to run")
            return
        
        # Get available GPU pairs
        available_gpu_pairs = self.gpu_detector.get_gpu_pairs(len(pending_tasks))
        
        if not available_gpu_pairs:
            logger.error("No available GPU pairs found!")
            return
        
        # Limit tasks to available GPU pairs
        tasks_to_run = pending_tasks[:len(available_gpu_pairs)]
        logger.info(f"Running {len(tasks_to_run)} tasks with {len(available_gpu_pairs)} GPU pairs")
        
        # Start workers
        threads = []
        for i, (low, high) in enumerate(tasks_to_run):
            gpu_ids = available_gpu_pairs[i]
            
            def worker_thread(l, h, gpus):
                return self.worker_manager.run_worker(l, h, gpus)
            
            thread = threading.Thread(
                target=worker_thread,
                args=(low, high, gpu_ids),
                name=f"Worker-{low}-{high}"
            )
            
            self.task_tracker.mark_task_running(low, high, thread)
            thread.start()
            threads.append(thread)
        
        # Monitor progress
        self.monitor_progress(threads, tasks_to_run)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
    
    def monitor_progress(self, threads: List[threading.Thread], tasks: List[Tuple[int, int]]):
        """Monitor task progress."""
        start_time = time.time()
        
        while any(thread.is_alive() for thread in threads):
            running_count = sum(1 for thread in threads if thread.is_alive())
            completed_count = len(tasks) - running_count
            
            elapsed_time = time.time() - start_time
            logger.info(f"Progress: {completed_count}/{len(tasks)} completed, "
                       f"{running_count} running, elapsed: {elapsed_time:.1f}s")
            
            time.sleep(30)  # Check every 30 seconds
    
    def merge_results(self):
        """Merge all result files into a single file."""
        logger.info("Starting result merge...")
        
        result_files = list(self.output_path.glob("generation_results_data_*_to_*_real.csv"))
        
        if not result_files:
            logger.warning("No result files found to merge")
            return
        
        all_dataframes = []
        
        for result_file in result_files:
            try:
                df = pd.read_csv(result_file)
                if len(df) > 0:
                    all_dataframes.append(df)
                    logger.info(f"Loaded {len(df)} rows from {result_file.name}")
            except Exception as e:
                logger.error(f"Error reading {result_file}: {e}")
        
        if all_dataframes:
            merged_df = pd.concat(all_dataframes, ignore_index=True)
            
            # Sort by data_id and token_id for consistency
            if 'data_id' in merged_df.columns and 'token_id' in merged_df.columns:
                merged_df = merged_df.sort_values(['data_id', 'token_id'])
            
            merged_file = self.output_path / "generation_results_data_all_real_merged.csv"
            merged_df.to_csv(merged_file, index=False)
            
            logger.info(f"Merged {len(all_dataframes)} files into {merged_file}")
            logger.info(f"Total rows in merged file: {len(merged_df)}")
            
            summary = {
                'total_files_merged': len(all_dataframes),
                'total_rows': len(merged_df),
                'merge_timestamp': datetime.now().isoformat(),
                'individual_files': [f.name for f in result_files if f.stat().st_size > 0]
            }
            
            # Make sure summary is JSON serializable
            serializable_summary = make_json_serializable(summary)
            
            with open(self.output_path / 'merge_summary.json', 'w') as f:
                json.dump(serializable_summary, f, indent=2)
        else:
            logger.error("No valid result files found to merge!")
    
    def run(self):
        """Main execution method."""
        logger.info("=== Starting Automatic LLM Continuation Scheduler ===")
        
        # Load input data
        logger.info(f"Loading input data from {self.args.input_path}")
        input_data = pd.read_csv(self.args.input_path)
        
        # Get unique data_ids
        unique_data_ids = sorted(input_data['data_id'].unique())
        total_samples = len(unique_data_ids)
        min_data_id, max_data_id = min(unique_data_ids), max(unique_data_ids)
        
        logger.info(f"Found {total_samples} unique data samples (IDs: {min_data_id} to {max_data_id})")
        
        # Split data into ranges
        index_ranges = self.split_data_ranges(total_samples, self.args.num_workers)
        
        # Convert to actual data_id ranges
        actual_ranges = []
        for low_idx, high_idx in index_ranges:
            actual_low = unique_data_ids[low_idx] if low_idx < len(unique_data_ids) else unique_data_ids[-1]
            actual_high = unique_data_ids[high_idx-1] + 1 if high_idx <= len(unique_data_ids) else unique_data_ids[-1] + 1
            actual_ranges.append((actual_low, actual_high))
            logger.info(f"Task range: data_id [{actual_low}, {actual_high}) ({high_idx-low_idx} samples)")
        
        # Get pending tasks
        pending_tasks = self.get_pending_tasks(actual_ranges)
        
        if not pending_tasks:
            logger.info("All tasks already completed!")
        else:
            # Run tasks in batches if there are more tasks than available GPUs
            while pending_tasks:
                current_batch = pending_tasks[:self.args.max_parallel_workers]
                pending_tasks = pending_tasks[self.args.max_parallel_workers:]
                
                logger.info(f"Starting batch with {len(current_batch)} tasks")
                self.run_tasks(current_batch)
                
                if pending_tasks:
                    logger.info(f"Waiting before next batch ({len(pending_tasks)} tasks remaining)")
                    time.sleep(10)  # Brief pause between batches
        
        self.merge_results()
        
        logger.info("=== Scheduler completed successfully ===")
    
    def cleanup(self):
        logger.info(f"Cleaning up scheduler {self.scheduler_id} resources...")
        self.worker_manager.cleanup()
        
        # Release any allocated GPUs
        if hasattr(self.gpu_detector, 'allocated_gpus'):
            released_gpus = list(self.gpu_detector.allocated_gpus)
            self.gpu_detector.allocated_gpus.clear()
            if released_gpus:
                logger.info(f"Released GPUs: {released_gpus}")
        
        logger.info("Scheduler cleanup completed")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Automatic LLM Continuation Scheduler')
    
    # Required arguments
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input CSV file from step 1')
    parser.add_argument('--output_path', type=str, default='output/playground/continuation_auto',
                        help='Path to store output files')
    parser.add_argument('--num_workers', type=int, required=True,
                        help='Total number of workers to create')
    
    # GPU configuration
    parser.add_argument('--gpus_per_worker', type=int, default=2,
                        help='Number of GPUs per worker')
    parser.add_argument('--min_gpu_memory', type=float, default=10.0,
                        help='Minimum free GPU memory (GB) required')
    parser.add_argument('--max_parallel_workers', type=int, default=8,
                        help='Maximum number of workers to run simultaneously')
    
    # Model configuration
    parser.add_argument('--tp_size', type=int, default=2,
                        help='Tensor parallelism size')
    parser.add_argument('--dp_size', type=int, default=1,
                        help='Data parallelism size')
    
    # Model parameters
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
    parser.add_argument('--show_worker_output', action='store_true', default=True,
                        help='Whether to show real-time worker output')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='Quiet mode - hide worker output (same as --no-show_worker_output)')
    
    args = parser.parse_args()
    
    if args.quiet:
        args.show_worker_output = False
    
    return args


def main():
    """Main function."""
    args = parse_args()
    
    if args.num_workers <= 0:
        raise ValueError("num_workers must be positive")
    
    if not Path(args.input_path).exists():
        raise FileNotFoundError(f"Input file not found: {args.input_path}")
    
    scheduler = AutoScheduler(args)
    
    try:
        scheduler.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        scheduler.cleanup()
    except Exception as e:
        logger.error(f"Scheduler failed: {e}")
        scheduler.cleanup()
        raise


if __name__ == "__main__":
    main() 