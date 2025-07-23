import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ.setdefault('MASTER_PORT', '29500')
import time
import argparse
import torch
import multiprocessing as mp
import sglang as sgl
from transformers import AutoTokenizer
import warnings
import json
import statistics
from datetime import datetime
from r2r.models.dynamic_sglang_selector import DynamicSimpleSGLangSelector
from r2r.utils.config import (
    QUICK_COLOR, REFERENCE_COLOR, RESET, TOTAL_GPU_NUM,
    MODEL_DICT
)

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
torch.set_warn_always(False)

class PerformanceTimer:
    def __init__(self):
        self.start_event = None
        self.end_event = None
        self.start_time_cpu = None
        self.elapsed_time_s = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.start_time_cpu = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available() and self.start_event and self.end_event:
            self.end_event.record()
            torch.cuda.synchronize()
            self.elapsed_time_s = self.start_event.elapsed_time(self.end_event) / 1000.0
        elif self.start_time_cpu is not None:
            end_time_cpu = time.perf_counter()
            self.elapsed_time_s = end_time_cpu - self.start_time_cpu

    def get_elapsed_time(self):
        return self.elapsed_time_s

def get_test_prompts():
    """Return a list of test prompts for benchmarking"""
    return [
        "What is the capital of France?",
        "Explain the concept of machine learning in simple terms.",
        "Write a short story about a robot learning to paint.",
        "How do neural networks work?",
        "What are the main differences between Python and JavaScript?",
        "Describe the process of photosynthesis.",
        "What is the theory of relativity?",
        "How does blockchain technology work?",
        "Explain the concept of recursion in programming.",
        "What are the benefits of renewable energy?"
    ]

def get_diverse_test_prompts(count=128):
    """Return a large list of diverse test prompts for batch benchmarking"""
    base_prompts = [
        "What is the capital of {}?",
        "Explain {} in simple terms.",
        "Write a short story about {}.",
        "How does {} work?",
        "What are the main differences between {} and {}?",
        "Describe the process of {}.",
        "What is the theory of {}?",
        "How does {} technology work?",
        "Explain the concept of {} in {}.",
        "What are the benefits of {}?",
        "Compare {} with {}.",
        "What is the history of {}?",
        "How to implement {} in Python?",
        "What are the challenges of {}?",
        "Analyze the impact of {} on society.",
        "What is the future of {}?",
        "How to optimize {}?",
        "What are the principles of {}?",
        "Explain {} algorithm step by step.",
        "What are the applications of {}?"
    ]
    
    subjects = [
        "artificial intelligence", "quantum computing", "machine learning", "deep learning",
        "blockchain", "cryptocurrency", "neural networks", "computer vision", "natural language processing",
        "robotics", "autonomous vehicles", "renewable energy", "solar power", "wind energy",
        "climate change", "global warming", "photosynthesis", "genetic engineering", "CRISPR",
        "space exploration", "mars colonization", "black holes", "quantum mechanics", "relativity",
        "programming", "algorithms", "data structures", "databases", "web development",
        "mobile development", "cloud computing", "DevOps", "microservices", "containerization",
        "distributed systems", "scalability", "performance optimization", "security", "cryptography",
        "cybersecurity", "privacy", "ethics", "philosophy", "psychology", "sociology",
        "economics", "finance", "marketing", "business strategy", "entrepreneurship",
        "innovation", "creativity", "leadership", "teamwork", "communication", "productivity"
    ]
    
    programming_languages = ["Python", "JavaScript", "Java", "C++", "Go", "Rust", "TypeScript", "Swift"]
    technologies = ["React", "Vue", "Angular", "Docker", "Kubernetes", "MongoDB", "PostgreSQL", "Redis"]
    
    prompts = []
    
    # Generate diverse prompts by combining templates with subjects
    for i in range(count):
        template = base_prompts[i % len(base_prompts)]
        
        if "{}" in template:
            # Count number of placeholders
            placeholder_count = template.count("{}")
            
            if placeholder_count == 1:
                subject = subjects[i % len(subjects)]
                prompt = template.format(subject)
            elif placeholder_count == 2:
                if "differences between" in template:
                    lang1 = programming_languages[i % len(programming_languages)]
                    lang2 = programming_languages[(i + 1) % len(programming_languages)]
                    prompt = template.format(lang1, lang2)
                elif "Compare" in template:
                    tech1 = technologies[i % len(technologies)]
                    tech2 = technologies[(i + 1) % len(technologies)]
                    prompt = template.format(tech1, tech2)
                else:
                    subject1 = subjects[i % len(subjects)]
                    subject2 = subjects[(i + 7) % len(subjects)]  # Use offset to get different subject
                    prompt = template.format(subject1, subject2)
            else:
                prompt = template
        else:
            prompt = template
            
        # Add some variation to make prompts more diverse
        if i % 10 == 0:
            prompt = f"In detail, {prompt.lower()}"
        elif i % 7 == 0:
            prompt = f"Briefly, {prompt.lower()}"
        elif i % 5 == 0:
            prompt = f"From a technical perspective, {prompt.lower()}"
            
        prompts.append(prompt)
    
    return prompts

class ModelBenchmark:
    def __init__(self, args):
        self.args = args
        self.test_prompts = get_test_prompts()
        self.batch_prompts = get_diverse_test_prompts(args.batch_size) if args.test_batch else []
        self.results = {}
        
    def _init_model(self, model_type: str):
        """Initialise the underlying generation engine / tokenizer according to model_type.

        Returns
        -------
        engine : The generation object (sgl.Engine or DynamicSimpleSGLangSelector)
        tokenizer : Tokenizer corresponding to the model
        is_r2r : bool, whether the returned engine is an R2R selector
        """
        model_type = model_type.lower()
        if model_type == "slm" or model_type == "llm":
            if model_type == "slm":
                model_path = MODEL_DICT['quick']['model_path']
                curr_tp_size = 1
            else:
                model_path = MODEL_DICT['reference']['model_path']
                curr_tp_size = self.args.tp_size
            try:
                engine = sgl.Engine(
                    model_path=model_path,
                    tp_size=curr_tp_size,
                    skip_tokenizer_init=True,
                    disable_radix_cache=True,
                    disable_cuda_graph=True,
                    dtype="bfloat16"
                )
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            except Exception as e:
                print(f"Failed to initialise SLM/LLM: {e}")
                return None, None, False
            return engine, tokenizer, False
        elif model_type == "r2r":
            if not self.args.router_path:
                print("Error: --router_path must be provided for R2R model type")
                return None, None, True

            strategy_kwargs = {'model_path': self.args.router_path}
            if self.args.neural_threshold:
                strategy_kwargs['threshold'] = self.args.neural_threshold

            sglang_kwargs = {
                "dtype": "bfloat16",
                "tp_size": self.args.tp_size
            }
            try:
                engine = DynamicSimpleSGLangSelector(
                    device="cuda",
                    dtype=torch.bfloat16,
                    switching_strategy='neural',
                    strategy_kwargs=strategy_kwargs,
                    is_record=False,
                    sglang_kwargs=sglang_kwargs
                )
                tokenizer = engine.tokenizer
            except Exception as e:
                print(f"Failed to initialise R2R generator: {e}")
                return None, None, True
            return engine, tokenizer, True
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def _prepare_batch_inputs(self, prompts, tokenizer):
        """Helper to convert text prompts to token ids for batch generation."""
        current_turn_messages_list = [{"role": "user", "content": prompt} for prompt in prompts]
        prompt_texts = [
            tokenizer.apply_chat_template(
                [current_turn_messages],
                tokenize=False,
                add_generation_prompt=True
            )
            for current_turn_messages in current_turn_messages_list
        ]
        batch_input_tokens = tokenizer.batch_encode_plus(prompt_texts, return_tensors=None)["input_ids"]
        return batch_input_tokens

    def evaluate(self, model_type: str, use_batch: bool):
        """Unified benchmark function.

        Parameters
        ----------
        model_type : str
            One of {"slm", "llm", "r2r"}.
        use_batch : bool
            Whether to evaluate with batch generation.
        """
        header_map = {
            'slm': f"\n{QUICK_COLOR}=== Benchmarking SLM{' - Batch' if use_batch else ''} ==={RESET}",
            'llm': f"\n{REFERENCE_COLOR}=== Benchmarking LLM{' - Batch' if use_batch else ''} ==={RESET}",
            'r2r': f"\n=== Benchmarking R2R Dynamic Mode{' - Batch' if use_batch else ''} ===",
        }
        print(header_map.get(model_type.lower(), f"\n=== Benchmarking {model_type} ==="))

        engine, tokenizer, is_r2r = self._init_model(model_type)
        if engine is None or tokenizer is None:
            return None

        prompts = self.batch_prompts if use_batch else self.test_prompts
        sampling_params_dict = {
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "max_new_tokens": self.args.max_new_tokens,
        }
        sampling_params_dict = {k: v for k, v in sampling_params_dict.items() if v is not None}

        # =================== Batch evaluation ===================
        if use_batch:
            batch_input_tokens = self._prepare_batch_inputs(prompts, tokenizer)
            print(f"Preparing batch of {len(prompts)} prompts...")

            print("Warming up...")
            if is_r2r:
                _ = engine.generate(
                    batch_input_tokens,
                    max_new_tokens=100,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    record_generation=False,
                    print_tokens=False,
                )
            else:
                new_sampling_params_dict = {k: v for k, v in sampling_params_dict.items() if v is not None}
                new_sampling_params_dict['max_new_tokens'] = 10
                _ = engine.generate(
                    input_ids=batch_input_tokens,
                    sampling_params=new_sampling_params_dict,
                    stream=False,
                )

            print("Warmup complete. Starting batch evaluation...")

            with PerformanceTimer() as timer:
                if is_r2r:
                    generated_texts, _ = engine.generate(
                        batch_input_tokens,
                        max_new_tokens=self.args.max_new_tokens,
                        temperature=self.args.temperature,
                        top_p=self.args.top_p,
                        record_generation=True,
                        print_tokens=False,
                    )
                else:
                    generated_stream = engine.generate(
                        input_ids=batch_input_tokens,
                        sampling_params=sampling_params_dict,
                        stream=False,
                    )
                    generated_results = list(generated_stream)

            elapsed_time_s = timer.get_elapsed_time()

            # ------------------ Post-processing ------------------
            total_output_tokens = 0
            generated_texts_processed = []

            if is_r2r:
                for i, text in enumerate(generated_texts):
                    # Exclude prompt tokens from output token count
                    num_output_tokens = len(tokenizer.encode(text)) - len(batch_input_tokens[i])
                    total_output_tokens += num_output_tokens
                    # Strip prompt from generated text for readability
                    processed_text = tokenizer.decode(
                        tokenizer.encode(text)[len(batch_input_tokens[i]):],
                        skip_special_tokens=True,
                    )
                    generated_texts_processed.append(processed_text)
            else:
                for res in generated_results:
                    text = tokenizer.decode(res['output_ids'], skip_special_tokens=True)
                    generated_texts_processed.append(text)
                    num_output_tokens = len(tokenizer.encode(text))
                    total_output_tokens += num_output_tokens

            tokens_per_second = total_output_tokens / elapsed_time_s if elapsed_time_s > 0 else 0

            result = {
                'batch_size': len(prompts),
                'total_output_tokens': total_output_tokens,
                'total_time_s': elapsed_time_s,
                'tokens_per_s': tokens_per_second,
                'avg_tokens_per_request': total_output_tokens / len(prompts),
                'generated_texts': generated_texts_processed,
            }

            # Logging summary
            print(f"Batch completed: {total_output_tokens} tokens in {elapsed_time_s:.2f}s")
            print(f"Speed: {tokens_per_second:.2f} tok/s")
            print(f"Average tokens per request: {result['avg_tokens_per_request']:.2f}")

            # Clean up
            if hasattr(engine, 'shutdown'):
                engine.shutdown()
            return result

        # =================== Single-prompt evaluation ===================
        results = []
        for i, prompt in enumerate(prompts):
            print(f"Testing prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            current_turn_messages = [{"role": "user", "content": prompt}]
            prompt_text = tokenizer.apply_chat_template(
                current_turn_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_ids = [tokenizer.encode(prompt_text)]  # Always wrap in list to align with engine API

            with PerformanceTimer() as timer:
                if is_r2r:
                    generated_texts, _ = engine.generate(
                        input_ids,
                        max_new_tokens=self.args.max_new_tokens,
                        temperature=self.args.temperature,
                        top_p=self.args.top_p,
                        record_generation=False,
                        print_tokens=False,
                    )
                    generated_text_full = generated_texts[0]
                else:
                    generated_stream = engine.generate(
                        input_ids=input_ids,
                        sampling_params=sampling_params_dict,
                        stream=False,
                    )
                    batch_res = list(generated_stream)
                    generated_text_full = tokenizer.decode(
                        batch_res[0]['output_ids'], skip_special_tokens=True
                    )

            elapsed_time_s = timer.get_elapsed_time()

            if is_r2r:
                # Strip prompt from generated part
                output_text = tokenizer.decode(
                    tokenizer.encode(generated_text_full)[len(input_ids[0]):],
                    skip_special_tokens=True,
                )
                num_output_tokens = len(tokenizer.encode(generated_text_full)) - len(input_ids[0])
            else:
                output_text = generated_text_full
                num_output_tokens = len(tokenizer.encode(output_text))

            tokens_per_second = num_output_tokens / elapsed_time_s if elapsed_time_s > 0 else 0

            record = {
                'prompt': prompt,
                'output_tokens': num_output_tokens,
                'time_s': elapsed_time_s,
                'tokens_per_s': tokens_per_second,
                'generated_text': output_text,
            }

            results.append(record)

            # Logging similar to original implementation
            print(f"  Output tokens: {num_output_tokens}, Time: {elapsed_time_s:.2f}s, Speed: {tokens_per_second:.2f} tok/s")

        if hasattr(engine, 'shutdown'):
            engine.shutdown()
        return results

    def run_all_benchmarks(self):
        """Run all benchmarks using unified evaluate method"""
        print("Starting Speed Benchmark...")
        print(f"Test prompts: {len(self.test_prompts)}")
        print(f"Max new tokens: {self.args.max_new_tokens}")
        print(f"Temperature: {self.args.temperature}")
        print(f"Top-p: {self.args.top_p}")

        if self.args.test_batch:
            print(f"Batch size: {self.args.batch_size}")

        # Mapping from flag to model_type string used by our unified evaluate
        flag_to_model = [
            (self.args.test_slm, 'SLM', 'slm'),
            (self.args.test_llm, 'LLM', 'llm'),
            (self.args.test_r2r, 'R2R', 'r2r'),
        ]

        for flag, display_name, model_key in flag_to_model:
            if not flag:
                continue
            suffix = ' - Batch' if self.args.test_batch else ''
            result_key = f"{display_name}{suffix}"
            self.results[result_key] = self.evaluate(model_key, use_batch=self.args.test_batch)

        self.print_summary()
        self.save_results()

    def print_summary(self):
        """Print benchmark summary"""
        print(f"\n{'='*60}")
        print(f"BENCHMARK SUMMARY")
        print(f"{'='*60}")
        
        for model_name, results in self.results.items():
            if results is None:
                print(f"\n{model_name}: FAILED")
                continue
            
            # Check if this is a batch result or individual results
            if isinstance(results, dict) and 'batch_size' in results:
                # This is a batch result
                print(f"\n{model_name}:")
                print(f"  Batch size: {results['batch_size']}")
                print(f"  Total tokens: {results['total_output_tokens']}")
                print(f"  Total time: {results['total_time_s']:.2f}s")
                print(f"  Speed: {results['tokens_per_s']:.2f} tok/s")
                print(f"  Average tokens per request: {results['avg_tokens_per_request']:.2f}")
            else:
                # This is individual results (list of results)
                total_tokens = sum(r['output_tokens'] for r in results)
                total_time = sum(r['time_s'] for r in results)
                avg_speed = statistics.mean([r['tokens_per_s'] for r in results])
                min_speed = min([r['tokens_per_s'] for r in results])
                max_speed = max([r['tokens_per_s'] for r in results])
                
                print(f"\n{model_name}:")
                print(f"  Total tokens: {total_tokens}")
                print(f"  Total time: {total_time:.2f}s")
                print(f"  Average speed: {avg_speed:.2f} tok/s")
                print(f"  Speed range: {min_speed:.2f} - {max_speed:.2f} tok/s")
        
        # Speed comparison
        if len(self.results) > 1:
            print(f"\nSPEED COMPARISON:")
            speeds = {}
            for model_name, results in self.results.items():
                if results:
                    if isinstance(results, dict) and 'batch_size' in results:
                        # Batch result
                        speeds[model_name] = results['tokens_per_s']
                    else:
                        # Individual results
                        speeds[model_name] = statistics.mean([r['tokens_per_s'] for r in results])
            
            sorted_speeds = sorted(speeds.items(), key=lambda x: x[1], reverse=True)
            for i, (model_name, speed) in enumerate(sorted_speeds):
                print(f"  {i+1}. {model_name}: {speed:.2f} tok/s")
    
    def save_results(self):
        """Save detailed results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"speed_benchmark_{timestamp}.json"
        
        # Save in current working directory
        filepath = os.path.join(os.getcwd(), filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Speed Benchmark: R2R vs SLM vs LLM")
    
    # Model selection
    parser.add_argument('--test_slm', action='store_true', default=False,
                        help='Test SLM speed')
    parser.add_argument('--test_llm', action='store_true', default=False,
                        help='Test LLM speed')
    parser.add_argument('--test_r2r', action='store_true', default=False,
                        help='Test R2R dynamic mode speed')
    
    # Batch testing
    parser.add_argument('--test_batch', action='store_true', default=False,
                        help='Test batch processing with large batch sizes')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for batch testing (default: 128)')
    
    # R2R configuration
    parser.add_argument('--router_path', type=str, default=None,
                        help='Path to the critical classifier model for R2R mode')
    parser.add_argument('--neural_threshold', type=float, default=0.9,
                        help='Threshold for the neural switching strategy')
    
    # Hardware configuration
    parser.add_argument('--tp_size', type=int, default=TOTAL_GPU_NUM if TOTAL_GPU_NUM > 0 else 1,
                        help='Tensor parallelism size for SGLang')
    
    # Generation parameters
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=1.0,
                        help='Top-p for nucleus sampling')
    parser.add_argument('--max_new_tokens', type=int, default=100,
                        help='Maximum new tokens to generate (default: 100 for quick testing)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.test_r2r and not args.router_path:
        print("Warning: --router_path is required for R2R testing. Skipping R2R benchmark.")
        args.test_r2r = False
    
    if not (args.test_slm or args.test_llm or args.test_r2r):
        print("Error: At least one test mode must be enabled")
        return
    
    if args.test_batch and args.batch_size <= 0:
        print("Error: batch_size must be positive when batch testing is enabled")
        return
    
    if args.test_batch and not (args.test_slm or args.test_llm or args.test_r2r):
        print("Error: At least one model test must be enabled for batch testing")
        return
    
    benchmark = ModelBenchmark(args)
    benchmark.run_all_benchmarks()

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"CUDA available with {torch.cuda.device_count()} GPUs")
    else:
        print("WARNING: CUDA not available")

    mp.set_start_method("spawn", force=True)
    main() 