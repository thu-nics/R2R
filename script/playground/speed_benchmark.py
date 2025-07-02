import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
import sys
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

# Add the root directory to Python path for imports
# script_dir = os.path.dirname(os.path.abspath(__file__))
# root_dir = os.path.dirname(os.path.dirname(script_dir))
# sys.path.insert(0, root_dir)

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
        
    def run_slm_benchmark(self):
        """Benchmark Small Language Model (1.5B)"""
        print(f"\n{QUICK_COLOR}=== Benchmarking SLM (1.5B) ==={RESET}")
        
        model_path = MODEL_DICT['quick']['model_path']
        
        try:
            llm_engine = sgl.Engine(
                model_path=model_path,
                tp_size=1,
                skip_tokenizer_init=True,
                dtype="bfloat16"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            print(f"Failed to initialize SLM: {e}")
            return None
            
        results = []
        
        for i, prompt in enumerate(self.test_prompts):
            print(f"Testing prompt {i+1}/{len(self.test_prompts)}: {prompt[:50]}...")
            
            current_turn_messages = [{"role": "user", "content": prompt}]
            prompt_text = tokenizer.apply_chat_template(
                current_turn_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            input_tokens = [tokenizer.encode(prompt_text)]
            
            sampling_params_dict = {
                "temperature": self.args.temperature,
                "top_p": self.args.top_p,
                "max_new_tokens": self.args.max_new_tokens,
            }
            sampling_params_dict = {k: v for k, v in sampling_params_dict.items() if v is not None}
            
            full_generated_text = ""
            
            with PerformanceTimer() as timer:
                generated_stream = llm_engine.generate(
                    input_ids=input_tokens, 
                    sampling_params=sampling_params_dict,
                    stream=True
                )
                
                for output_item in generated_stream:
                    output_text = tokenizer.decode(output_item['output_ids'][-1:], skip_special_tokens=True)
                    full_generated_text += output_text
            
            elapsed_time_s = timer.get_elapsed_time()
            num_output_tokens = len(tokenizer.encode(full_generated_text))
            tokens_per_second = num_output_tokens / elapsed_time_s if elapsed_time_s > 0 else 0
            
            results.append({
                'prompt': prompt,
                'output_tokens': num_output_tokens,
                'time_s': elapsed_time_s,
                'tokens_per_s': tokens_per_second,
                'generated_text': full_generated_text
            })
            
            print(f"  Output tokens: {num_output_tokens}, Time: {elapsed_time_s:.2f}s, Speed: {tokens_per_second:.2f} tok/s")
        
        llm_engine.shutdown()
        return results
    
    def run_llm_benchmark(self):
        """Benchmark Large Language Model (32B)"""
        print(f"\n{REFERENCE_COLOR}=== Benchmarking LLM (32B) ==={RESET}")
        
        model_path = MODEL_DICT['reference']['model_path']
        
        try:
            llm_engine = sgl.Engine(
                model_path=model_path,
                tp_size=self.args.tp_size,
                skip_tokenizer_init=True,
                dtype="bfloat16"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            print(f"Failed to initialize LLM: {e}")
            return None
            
        results = []
        
        for i, prompt in enumerate(self.test_prompts):
            print(f"Testing prompt {i+1}/{len(self.test_prompts)}: {prompt[:50]}...")
            
            current_turn_messages = [{"role": "user", "content": prompt}]
            prompt_text = tokenizer.apply_chat_template(
                current_turn_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            input_tokens = [tokenizer.encode(prompt_text)]
            
            sampling_params_dict = {
                "temperature": self.args.temperature,
                "top_p": self.args.top_p,
                "max_new_tokens": self.args.max_new_tokens,
            }
            sampling_params_dict = {k: v for k, v in sampling_params_dict.items() if v is not None}
            
            full_generated_text = ""
            
            with PerformanceTimer() as timer:
                generated_stream = llm_engine.generate(
                    input_ids=input_tokens, 
                    sampling_params=sampling_params_dict,
                    stream=True
                )
                
                for output_item in generated_stream:
                    output_text = tokenizer.decode(output_item['output_ids'][-1:], skip_special_tokens=True)
                    full_generated_text += output_text
            
            elapsed_time_s = timer.get_elapsed_time()
            num_output_tokens = len(tokenizer.encode(full_generated_text))
            tokens_per_second = num_output_tokens / elapsed_time_s if elapsed_time_s > 0 else 0
            
            results.append({
                'prompt': prompt,
                'output_tokens': num_output_tokens,
                'time_s': elapsed_time_s,
                'tokens_per_s': tokens_per_second,
                'generated_text': full_generated_text
            })
            
            print(f"  Output tokens: {num_output_tokens}, Time: {elapsed_time_s:.2f}s, Speed: {tokens_per_second:.2f} tok/s")
        
        llm_engine.shutdown()
        return results
    
    def run_r2r_benchmark(self):
        """Benchmark R2R Dynamic Mode"""
        print(f"\n=== Benchmarking R2R Dynamic Mode ===")
        
        if not self.args.router_path:
            print("Error: router_path is required for R2R benchmark")
            return None
            
        strategy_kwargs = {
            'model_path': self.args.router_path
        }
        if self.args.neural_threshold:
            strategy_kwargs['threshold'] = self.args.neural_threshold

        ref_model_path = MODEL_DICT['reference']['model_path']
        qck_model_path = MODEL_DICT['quick']['model_path']

        sglang_kwargs = {
            "dtype": "bfloat16",
            "tp_size": self.args.tp_size
        }

        try:
            generator = DynamicSimpleSGLangSelector(
                device="cuda", 
                dtype=torch.bfloat16,
                switching_strategy='neural',
                strategy_kwargs=strategy_kwargs,
                is_record=True, 
                sglang_kwargs=sglang_kwargs
            )
        except Exception as e:
            print(f"Failed to initialize R2R generator: {e}")
            return None
            
        results = []
        
        for i, prompt in enumerate(self.test_prompts):
            print(f"Testing prompt {i+1}/{len(self.test_prompts)}: {prompt[:50]}...")
            
            current_turn_messages = [{"role": "user", "content": prompt}]
            prompt_text = generator.tokenizer.apply_chat_template(
                current_turn_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            input_ids = [generator.tokenizer.encode(prompt_text)]
            
            with PerformanceTimer() as timer:
                generated_texts, recorders = generator.generate(
                    input_ids,
                    max_new_tokens=self.args.max_new_tokens,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    record_generation=True, 
                    print_tokens=False
                )
            
            elapsed_time_s = timer.get_elapsed_time()
            generated_text_full = generated_texts[0]
            
            num_output_tokens = len(generator.tokenizer.encode(generated_text_full)) - len(input_ids[0])
            tokens_per_second = num_output_tokens / elapsed_time_s if elapsed_time_s > 0 else 0
            
            # Get model usage statistics from recorder
            recorder = recorders[0]
            quick_tokens = sum(1 for record in recorder.records if record.source_model == 'quick')
            reference_tokens = sum(1 for record in recorder.records if record.source_model == 'reference')
            
            results.append({
                'prompt': prompt,
                'output_tokens': num_output_tokens,
                'time_s': elapsed_time_s,
                'tokens_per_s': tokens_per_second,
                'generated_text': generated_text_full,
                'quick_tokens': quick_tokens,
                'reference_tokens': reference_tokens,
                'quick_ratio': quick_tokens / (quick_tokens + reference_tokens) if (quick_tokens + reference_tokens) > 0 else 0
            })
            
            print(f"  Output tokens: {num_output_tokens}, Time: {elapsed_time_s:.2f}s, Speed: {tokens_per_second:.2f} tok/s")
            print(f"  Quick: {quick_tokens}, Reference: {reference_tokens}, Quick ratio: {results[-1]['quick_ratio']:.2%}")
        
        if hasattr(generator, 'shutdown'):
            generator.shutdown()
        
        return results
    
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
                
                if 'quick_ratio' in results:
                    print(f"  Quick model usage: {results['quick_ratio']:.2%}")
                    print(f"  Quick tokens: {results['total_quick_tokens']}")
                    print(f"  Reference tokens: {results['total_reference_tokens']}")
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
                
                if model_name == "R2R" and results:
                    avg_quick_ratio = statistics.mean([r['quick_ratio'] for r in results])
                    print(f"  Average quick model usage: {avg_quick_ratio:.2%}")
        
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
    
    def run_all_benchmarks(self):
        """Run all benchmarks"""
        print("Starting Speed Benchmark...")
        print(f"Test prompts: {len(self.test_prompts)}")
        print(f"Max new tokens: {self.args.max_new_tokens}")
        print(f"Temperature: {self.args.temperature}")
        print(f"Top-p: {self.args.top_p}")
        
        if self.args.test_batch:
            print(f"Batch size: {self.args.batch_size}")
        
        # Run SLM benchmark
        if self.args.test_slm:
            if self.args.test_batch:
                self.results["SLM (1.5B) - Batch"] = self.run_slm_batch_benchmark()
            else:
                self.results["SLM (1.5B)"] = self.run_slm_benchmark()
        
        # Run LLM benchmark
        if self.args.test_llm:
            if self.args.test_batch:
                self.results["LLM (32B) - Batch"] = self.run_llm_batch_benchmark()
            else:
                self.results["LLM (32B)"] = self.run_llm_benchmark()
        
        # Run R2R benchmark
        if self.args.test_r2r:
            if self.args.test_batch:
                self.results["R2R - Batch"] = self.run_r2r_batch_benchmark()
            else:
                self.results["R2R"] = self.run_r2r_benchmark()
        
        self.print_summary()
        self.save_results()

    def run_slm_batch_benchmark(self):
        """Benchmark Small Language Model (1.5B) with batch processing"""
        print(f"\n{QUICK_COLOR}=== Benchmarking SLM (1.5B) - Batch Size {self.args.batch_size} ==={RESET}")
        
        model_path = MODEL_DICT['quick']['model_path']
        
        try:
            llm_engine = sgl.Engine(
                model_path=model_path,
                tp_size=1,
                skip_tokenizer_init=True,
                dtype="bfloat16"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            print(f"Failed to initialize SLM: {e}")
            return None
        
        print(f"Preparing batch of {len(self.batch_prompts)} diverse prompts...")
        
        # Prepare all prompts for batch processing
        batch_messages = []
        batch_input_tokens = []
        for prompt in self.batch_prompts:
            current_turn_messages = [{"role": "user", "content": prompt}]
            prompt_text = tokenizer.apply_chat_template(
                current_turn_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            batch_input_tokens.append(tokenizer.encode(prompt_text))
            batch_messages.append(current_turn_messages)
        
        sampling_params_dict = {
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "max_new_tokens": self.args.max_new_tokens,
        }
        sampling_params_dict = {k: v for k, v in sampling_params_dict.items() if v is not None}
        
        print(f"Starting batch generation...")
        
        with PerformanceTimer() as timer:
            generated_stream = llm_engine.generate(
                input_ids=batch_input_tokens, 
                sampling_params=sampling_params_dict,
                stream=False  # Disable streaming for batch processing
            )
            
            # Collect all results
            batch_results = list(generated_stream)
        
        elapsed_time_s = timer.get_elapsed_time()
        
        # Process results
        total_output_tokens = 0
        generated_texts = []
        
        for i, output_item in enumerate(batch_results):
            generated_text = tokenizer.decode(output_item['output_ids'][len(batch_input_tokens[i]):], skip_special_tokens=True)
            generated_texts.append(generated_text)
            num_output_tokens = len(tokenizer.encode(generated_text))
            total_output_tokens += num_output_tokens
        
        tokens_per_second = total_output_tokens / elapsed_time_s if elapsed_time_s > 0 else 0
        
        result = {
            'batch_size': len(self.batch_prompts),
            'total_output_tokens': total_output_tokens,
            'total_time_s': elapsed_time_s,
            'tokens_per_s': tokens_per_second,
            'avg_tokens_per_request': total_output_tokens / len(self.batch_prompts),
            'generated_texts': generated_texts
        }
        
        print(f"Batch completed: {total_output_tokens} tokens in {elapsed_time_s:.2f}s")
        print(f"Speed: {tokens_per_second:.2f} tok/s")
        print(f"Average tokens per request: {result['avg_tokens_per_request']:.2f}")
        
        llm_engine.shutdown()
        return result
    
    def run_llm_batch_benchmark(self):
        """Benchmark Large Language Model (32B) with batch processing"""
        print(f"\n{REFERENCE_COLOR}=== Benchmarking LLM (32B) - Batch Size {self.args.batch_size} ==={RESET}")
        
        model_path = MODEL_DICT['reference']['model_path']
        
        try:
            llm_engine = sgl.Engine(
                model_path=model_path,
                tp_size=self.args.tp_size,
                skip_tokenizer_init=True,
                dtype="bfloat16"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            print(f"Failed to initialize LLM: {e}")
            return None
        
        print(f"Preparing batch of {len(self.batch_prompts)} diverse prompts...")
        
        # Prepare all prompts for batch processing
        batch_messages = []
        batch_input_tokens = []
        for prompt in self.batch_prompts:
            current_turn_messages = [{"role": "user", "content": prompt}]
            prompt_text = tokenizer.apply_chat_template(
                current_turn_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            batch_input_tokens.append(tokenizer.encode(prompt_text))
            batch_messages.append(current_turn_messages)
        
        sampling_params_dict = {
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "max_new_tokens": self.args.max_new_tokens,
        }
        sampling_params_dict = {k: v for k, v in sampling_params_dict.items() if v is not None}
        
        print(f"Starting batch generation...")
        
        with PerformanceTimer() as timer:
            generated_stream = llm_engine.generate(
                input_ids=batch_input_tokens, 
                sampling_params=sampling_params_dict,
                stream=False  # Disable streaming for batch processing
            )
            
            # Collect all results
            batch_results = list(generated_stream)
        
        elapsed_time_s = timer.get_elapsed_time()
        
        # Process results
        total_output_tokens = 0
        generated_texts = []
        
        for i, output_item in enumerate(batch_results):
            generated_text = tokenizer.decode(output_item['output_ids'][len(batch_input_tokens[i]):], skip_special_tokens=True)
            generated_texts.append(generated_text)
            num_output_tokens = len(tokenizer.encode(generated_text))
            total_output_tokens += num_output_tokens
        
        tokens_per_second = total_output_tokens / elapsed_time_s if elapsed_time_s > 0 else 0
        
        result = {
            'batch_size': len(self.batch_prompts),
            'total_output_tokens': total_output_tokens,
            'total_time_s': elapsed_time_s,
            'tokens_per_s': tokens_per_second,
            'avg_tokens_per_request': total_output_tokens / len(self.batch_prompts),
            'generated_texts': generated_texts
        }
        
        print(f"Batch completed: {total_output_tokens} tokens in {elapsed_time_s:.2f}s")
        print(f"Speed: {tokens_per_second:.2f} tok/s")
        print(f"Average tokens per request: {result['avg_tokens_per_request']:.2f}")
        
        llm_engine.shutdown()
        return result
    
    def run_r2r_batch_benchmark(self):
        """Benchmark R2R Dynamic Mode with batch processing"""
        print(f"\n=== Benchmarking R2R Dynamic Mode - Batch Size {self.args.batch_size} ===")
        
        if not self.args.router_path:
            print("Error: router_path is required for R2R benchmark")
            return None
            
        strategy_kwargs = {
            'model_path': self.args.router_path
        }
        if self.args.neural_threshold:
            strategy_kwargs['threshold'] = self.args.neural_threshold

        ref_model_path = MODEL_DICT['reference']['model_path']
        qck_model_path = MODEL_DICT['quick']['model_path']

        sglang_kwargs = {
            "dtype": "bfloat16",
            "tp_size": self.args.tp_size
        }

        try:
            generator = DynamicSimpleSGLangSelector(
                device="cuda", 
                dtype=torch.bfloat16,
                switching_strategy='neural',
                strategy_kwargs=strategy_kwargs,
                is_record=True, 
                sglang_kwargs=sglang_kwargs
            )
        except Exception as e:
            print(f"Failed to initialize R2R generator: {e}")
            return None
        
        print(f"Preparing batch of {len(self.batch_prompts)} diverse prompts...")
        
        # Prepare all prompts for batch processing
        batch_messages = []
        batch_input_ids = []
        for prompt in self.batch_prompts:
            current_turn_messages = [{"role": "user", "content": prompt}]
            prompt_text = generator.tokenizer.apply_chat_template(
                current_turn_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            input_ids = generator.tokenizer.encode(prompt_text)
            batch_input_ids.append(input_ids)
            batch_messages.append(current_turn_messages)
        
        print(f"Starting batch generation...")
        
        with PerformanceTimer() as timer:
            generated_texts, recorders = generator.generate(
                batch_input_ids,
                max_new_tokens=self.args.max_new_tokens,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                record_generation=True, 
                print_tokens=False
            )
        
        elapsed_time_s = timer.get_elapsed_time()
        
        # Process results
        total_output_tokens = 0
        total_quick_tokens = 0
        total_reference_tokens = 0
        
        for i, (generated_text, recorder) in enumerate(zip(generated_texts, recorders)):
            num_output_tokens = len(generator.tokenizer.encode(generated_text)) - len(batch_input_ids[i])
            total_output_tokens += num_output_tokens
            
            # Get model usage statistics from recorder
            quick_tokens = sum(1 for record in recorder.records if record.source_model == 'quick')
            reference_tokens = sum(1 for record in recorder.records if record.source_model == 'reference')
            total_quick_tokens += quick_tokens
            total_reference_tokens += reference_tokens
        
        tokens_per_second = total_output_tokens / elapsed_time_s if elapsed_time_s > 0 else 0
        quick_ratio = total_quick_tokens / (total_quick_tokens + total_reference_tokens) if (total_quick_tokens + total_reference_tokens) > 0 else 0
        
        result = {
            'batch_size': len(self.batch_prompts),
            'total_output_tokens': total_output_tokens,
            'total_time_s': elapsed_time_s,
            'tokens_per_s': tokens_per_second,
            'avg_tokens_per_request': total_output_tokens / len(self.batch_prompts),
            'total_quick_tokens': total_quick_tokens,
            'total_reference_tokens': total_reference_tokens,
            'quick_ratio': quick_ratio,
            'generated_texts': generated_texts
        }
        
        print(f"Batch completed: {total_output_tokens} tokens in {elapsed_time_s:.2f}s")
        print(f"Speed: {tokens_per_second:.2f} tok/s")
        print(f"Average tokens per request: {result['avg_tokens_per_request']:.2f}")
        print(f"Quick: {total_quick_tokens}, Reference: {total_reference_tokens}, Quick ratio: {quick_ratio:.2%}")
        
        if hasattr(generator, 'shutdown'):
            generator.shutdown()
        
        return result

def main():
    parser = argparse.ArgumentParser(description="Speed Benchmark: R2R vs SLM vs LLM")
    
    # Model selection
    parser.add_argument('--test_slm', action='store_true', default=False,
                        help='Test SLM (1.5B) speed')
    parser.add_argument('--test_llm', action='store_true', default=False,
                        help='Test LLM (32B) speed')
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
    parser.add_argument('--neural_threshold', type=float, default=0.5,
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