import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ.setdefault('MASTER_PORT', '29500')
os.environ['SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK'] = '1'
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import pandas as pd
import json
from typing import Dict, List, Tuple, Callable, Any, Optional
import numpy as np
from datetime import datetime
import math
from tqdm import tqdm
import argparse
from r2r.models.dynamic_sglang_selector import DynamicSimpleSGLangSelector
from r2r.evaluate.eval_utils import get_answer_extractor, check_answer_correctness
from r2r.evaluate.eval_utils import QUERY_TEMPLATE_MULTICHOICE, ANSWER_PATTERN_MULTICHOICE
from r2r.evaluate.eval_utils import lcb_codegeneration_prompt_fn
import time
import sglang as sgl
from sglang.srt.hf_transformers_utils import get_tokenizer
from r2r.evaluate.eval_utils import select_by_category, generate_cot_prompt, preprocess
import multiprocessing as mp
import warnings
from r2r.utils.config import TOKEN_TYPE, MODEL_DICT

# set numpy random seed
np.random.seed(42)

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
torch.set_warn_always(False)

# Load dataset configurations from JSON file
def load_configs() -> Dict:
    dataset_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'evaluate/eval_configs/dataset_configs.json')
    model_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'evaluate/eval_configs/model_configs.json')
    try:
        with open(dataset_config_path, 'r') as f:
            dataset_configs = json.load(f)
        with open(model_config_path, 'r') as f:
            models = json.load(f)
        return dataset_configs, models
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset configuration file not found at {dataset_config_path}, {model_config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in dataset configuration file at {dataset_config_path}, {model_config_path}")

# Load r2r configurations from JSON file
def load_r2r_configs() -> Dict:
    r2r_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'evaluate/eval_configs/r2r_configs.json')
    try:
        with open(r2r_config_path, 'r') as f:
            r2r_configs = json.load(f)
        return r2r_configs
    except FileNotFoundError:
        print(f"Warning: R2R configuration file not found at {r2r_config_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON format in R2R configuration file at {r2r_config_path}")
        return {}

# Load dataset configurations
DATASET_CONFIGS, MODELS = load_configs()
R2R_CONFIGS = load_r2r_configs()

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate models on different datasets')
    
    # Model configuration
    parser.add_argument('--model_path', type=str, 
                      default='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
                      help='Path or name of the model to evaluate')
    parser.add_argument('--model_param', type=float, default=1.5,
                      help='Model parameter size in billions')
    parser.add_argument('--generator', type=str, default='sglang',
                      choices=['sglang'],
                      help='Generator for dynamic model selection')
    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='aime',
                      choices=list(DATASET_CONFIGS.keys()),
                      help=f'Dataset to use: {", ".join(DATASET_CONFIGS.keys())}')
    parser.add_argument('--dataset_path', type=str, default=None,
                      help='Override the default dataset path if needed')
    parser.add_argument('--dataset_config', type=str, default=None,
                      help='Dataset configuration name (e.g., "gpqa_diamond" for GPQA)')
    
    # Hardware configuration
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size per GPU')
    parser.add_argument('--use_hybrid', action='store_true', default=False,
                      help='Use hybrid model processing (default: False)')
    parser.add_argument('--mem_fraction_static', type=float, default=0.8,
                        help='Fraction of GPU memory to allocate for judging')
    # Path configuration
    parser.add_argument('--output_dir', type=str, default=None,
                      help='Directory to save results (defaults to output/{dataset}_eval)')
    
    # Generation configuration
    parser.add_argument('--max_new_tokens', type=int, default=32768,
                      help='Maximum number of new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.0,
                      help='Temperature for the model')
    parser.add_argument('--top_p', type=float, default=1.0,
                      help='Top-p for the model')
    parser.add_argument('--top_k', type=int, default=-1,
                    help='Top-k filtering parameter for sampling (default: -1)')
    parser.add_argument('--beam_size', type=int, default=3,
                      help='Beam size for tree-based generation')
    parser.add_argument('--tp_size', type=int, default=2, help='Number of tensor parallel GPUs')
    parser.add_argument('--dp_size', type=int, default=1, help='Number of data parallel GPUs')
    # Debug configuration
    parser.add_argument('--debug', action='store_true',
                      help='Run in debug mode (only process first problem)')
    parser.add_argument('--num_problems', type=int, default=None,
                      help='Number of problems to process (for testing)')
    
    # Recovery configuration
    parser.add_argument('--problem_ids', type=str, default=None,
                      help='Comma-separated list of specific problem IDs to process')
    parser.add_argument('--resume', action='store_true',
                      help='Resume from last checkpoint, processing only failed or missing problems')
    
    # Hybrid model configuration
    parser.add_argument('--switching_strategy', type=str, default='neural',
                      help='Switching strategy for hybrid model')
    parser.add_argument('--is_record', action='store_true',
                      help='Record hybrid model generation')
    
    # Neural router configuration
    parser.add_argument('--router_path', type=str, default='resource/default_router.pt',
                      help='Path to the neural router model')
    
    parser.add_argument('--threshold', type=float, default=None,
                      help='Threshold for neural router')
    
    parser.add_argument('--reference_prob', type=float, default=0.5,
                      help='Probability of selecting the reference model')
    
    parser.add_argument('--test_run_time', type=bool, default=True,
                      help='Test run time of the model')
    
    # Add job-based processing arguments
    parser.add_argument("--split_jobs", action="store_true",
                      help="Split jobs for distributed processing")
    parser.add_argument('--job_nums', type=int, default=1,
                      help='Total number of jobs for distributed processing (used when threads_per_gpu=0)')
    parser.add_argument('--job_id', type=int, default=0,
                      help='Current job ID (0 to job_nums-1, or -1 to only combine results)')
    
    # Add repeat input parameter
    parser.add_argument('--repeat_input_num', type=int, default=1,
                      help='Number of times to repeat each input in batch dimension')
    
    args = parser.parse_args()

    args.test_run_time = True
    
    if args.use_hybrid:
        if args.model_path == MODEL_DICT['quick']['model_path']:
            print(f"Using quick model: {args.model_path}")
        else:
            print(f"model path does not match the quick model path: {args.model_path} and {MODEL_DICT['quick']['model_path']}, use quick model instead")
            args.model_path = MODEL_DICT['quick']['model_path']
        
    # Get dataset config
    dataset_config = DATASET_CONFIGS[args.dataset]
    
    # Set default output directory based on dataset if not provided
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"output/eval/hf_dataset_sglang/{args.dataset}_eval_{timestamp}"
    
    # Use dataset_path from config if not overridden
    if args.dataset_path is None:
        args.dataset_path = dataset_config["path"]
    
    # Use dataset_config from config if not overridden and it exists
    if args.dataset_config is None and "dataset_config" in dataset_config:
        args.dataset_config = dataset_config["dataset_config"]
    
    # Store dataset config in args for easy access
    args.dataset_config_dict = dataset_config
    
    # Load router_path and threshold from r2r configs
    if args.dataset in R2R_CONFIGS:
        r2r_config = R2R_CONFIGS[args.dataset]
        # Handle router_path
        if 'router_path' in r2r_config:
            if args.router_path != r2r_config['router_path']:
                warnings.warn(
                    f"Router path mismatch for dataset '{args.dataset}': "
                    f"provided '{args.router_path}' but r2r config specifies '{r2r_config['router_path']}'"
                )
            else:
                print(f"Using provided router_path (matches r2r config): {args.router_path}")
        # Handle threshold (only load if not provided)
        if args.threshold is None and 'threshold' in r2r_config:
            args.threshold = r2r_config['threshold']
            print(f"Using threshold from r2r config: {args.threshold}")
    
    if args.split_jobs and 'job' not in args.output_dir and args.job_id >= 0:
        args.output_dir = os.path.join(args.output_dir, f'job_{args.job_id}')
    
    return args


def setup_device(args: argparse.Namespace, gpu_id: int, thread_id: int) -> Tuple[str, Any]:
    """Set up device(s) for model processing based on arguments.
    
    Returns:
        Tuple[str, Any]: A tuple containing (primary_device, device_map)
    """
    # Original single GPU behavior
    torch.cuda.set_device(gpu_id)
    primary_device = f"cuda:{gpu_id}"
    device_map = primary_device
    print(f"Setting up on GPU {gpu_id} (thread {thread_id})...")
    
    return primary_device, device_map

def process_problems(
    args: argparse.Namespace, 
    problems: List[Dict], 
    use_hybrid: bool
):
    """Process a set of problems using either standard or hybrid model.
    
    This function handles device setup, model initialization, and problem processing.
    It routes the processing to either standard or hybrid model based on use_hybrid flag.
    
    Args:
        args: Command line arguments
        problems: List of problems to process
        use_hybrid: Whether to use hybrid model processing
        
    Returns:
        List of result dictionaries containing model outputs and metrics
    """
    # Load tokenizer (common for both approaches)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Get dataset configuration
    dataset_config = args.dataset_config_dict
    answer_type = dataset_config.get("answer_type", "boxed")
    answer_extractor = get_answer_extractor(answer_type)

    # Process based on chosen approach
    results = process_with_model(
        args=args,
        tokenizer=tokenizer,
        problems=problems,
        answer_type=answer_type,
        answer_extractor=answer_extractor,
        dataset_config=dataset_config,
        use_hybrid=use_hybrid
    )
    
    # Add model metadata
    model_name = args.model_path.split('/')[-1]
    for result in results:
        result.update({
            "model_name": model_name,
            "model_params": args.model_param
        })
    
    # Save results
    save_results(results, model_name, -1, -1, args.output_dir)
    
    return results

def process_with_model(
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
    problems: List[Dict],
    answer_type: str,
    answer_extractor: Callable,
    dataset_config: Dict,
    use_hybrid: bool
) -> List[Dict]:
    """Process problems using either standard or hybrid model.
    
    Args:
        args: Command line arguments with model configuration
        tokenizer: Pre-loaded tokenizer for the model
        model: Initialized sglang Engine
        problems: List of problems to process
        answer_type: Type of answer extraction to use (boxed, multiple_choice)
        answer_extractor: Function to extract answers from model output
        dataset_config: Dataset-specific configuration
        use_hybrid: Whether to use hybrid model processing
    """
    generator = None
    kwargs_generation = dict()
    sampling_params = dict()
    if use_hybrid:
        model = None
        # Prepare strategy kwargs based on the selected strategy
        strategy_kwargs = {}
        if args.switching_strategy == 'rolling':
            strategy_kwargs.update({
                'window_size': args.window_size,
                'required_simple_ratio': args.required_simple_ratio
            })
        elif args.switching_strategy == 'random':
            strategy_kwargs.update({
                'reference_prob': args.reference_prob,
            })
        elif args.switching_strategy == 'momentum':
            strategy_kwargs.update({
                'momentum_factor': args.momentum_factor,
                'quick_to_ref_threshold': args.quick_to_ref_threshold,
                'ref_to_quick_threshold': args.ref_to_quick_threshold
            })
        elif args.switching_strategy == 'neural':
            strategy_kwargs.update({
                'model_path': args.router_path,
            })
            if args.threshold is not None:
                strategy_kwargs.update({
                    'threshold': args.threshold,
                })
        elif args.switching_strategy == 'neural_rolling':
            strategy_kwargs.update({
                'model_path': args.router_path,
                # 'threshold': 0.5,
                'window_size': args.window_size,
                'required_simple_ratio': args.required_simple_ratio,
            })
        elif args.switching_strategy == 'neural_multi_input':
            strategy_kwargs.update({
                'model_path': args.router_path,
                'threshold': 0.5,
                'neural_window_size': args.neural_window_size,
            })
        # initialize generator
        kwargs_init = dict()
        kwargs_generation = dict()

        if args.generator == "sglang":
            generator_class = DynamicSimpleSGLangSelector
            kwargs_init = {
                "sglang_kwargs": {
                    "dtype": "bfloat16",
                    "tp_size": args.tp_size,
                }
            }
            print(f"Using {args.tp_size} GPUs for SGLang")
        else:
            raise ValueError(f"Invalid generator: {args.generator}")
        generator = generator_class(
            device="cuda",
            dtype=torch.bfloat16,
            switching_strategy=args.switching_strategy,
            strategy_kwargs=strategy_kwargs,
            is_record=args.is_record,
            **kwargs_init
        )
    else:
        # Initialize sglang model with data parallelism
        print("Initializing model with data parallelism...")
        model = sgl.Engine(
            model_path=args.model_path, 
            dtype="bfloat16", 
            mem_fraction_static=args.mem_fraction_static,
            skip_tokenizer_init=False,
            tp_size=args.tp_size,
            dp_size=args.dp_size
        )
        sampling_params = {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            }
        
    results = evaluate_problem(
        args,
        model,
        tokenizer,
        problems,
        dataset_config,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        device='cuda',
        use_hybrid=use_hybrid,
        generator=generator,
        test_run_time=args.test_run_time,
        sampling_params=sampling_params,
        **kwargs_generation
    )
        
    return results

def evaluate_problem(
    args: argparse.Namespace,
    model: Any,
    tokenizer: AutoTokenizer,
    problems: List[Dict],
    dataset_config: Dict,
    max_new_tokens: int = 32768,
    batch_size: int = 1,
    device: str = "cuda",
    use_hybrid: bool = False,
    generator: DynamicSimpleSGLangSelector = None,
    test_run_time: bool = False,
    sampling_params: Dict = None,
    **kwargs_generation
) -> List[Dict]:
    """Evaluate a batch of problems using data parallelism."""
    # Check if repeat_input_num equals batch_size
    # if args.repeat_input_num > 1:   
    #     if args.repeat_input_num != batch_size:
    #         raise ValueError(f"repeat_input_num ({args.repeat_input_num}) must equal batch_size ({batch_size})")
    
    # Get the appropriate answer extractor for this dataset
    answer_type = dataset_config.get("answer_type", "boxed")
    answer_extractor = get_answer_extractor(answer_type)
    
    results = []
    run_time = None
    
    # Create temp directory for intermediate results
    temp_dir = os.path.join(args.output_dir, "temp")
    temp_csv_dir = os.path.join(args.output_dir, "temp_csv")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(temp_csv_dir, exist_ok=True)
    print(f"Saving intermediate results to {temp_dir}")
    print(f"Saving intermediate csv results to {temp_csv_dir}")
    # Track problem ID occurrences across batches
    problem_id_counts = {}
    
    # Repeat prompts if repeat_input_num > 1
    if args.repeat_input_num > 1:
        problems = [problem for problem in problems for _ in range(args.repeat_input_num)]
        print(f"Repeating problems {args.repeat_input_num} times")
    
    for i in tqdm(range(0, len(problems), batch_size), desc="Processing batches"):
        
        batch = problems[i:i + batch_size]
            
        # Prepare messages using chat template
        messages_list = [
            [{"role": "user", "content": item['FormattedProblem']}]
            for item in batch
        ]

        # Apply chat template to each message
        prompts = [
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in messages_list
        ]     

        # Process each item in batch
        if use_hybrid:
            # Generate with recording
            # inputs = [generator.tokenizer.encode(prompt)[1:] for prompt in prompts] # noqa: skip BOS token
            inputs = [generator.tokenizer.encode(prompt) for prompt in prompts]
            if test_run_time:
                start_time = time.time()
            generated_texts, recorders = generator.generate(
                inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                record_generation=True,
                print_tokens=False,
                **kwargs_generation
            )
            
            # End timer and print duration
            if test_run_time:
                end_time = time.time()
                run_time = end_time - start_time
        else:            
            # Tokenize inputs
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            input_lengths = [len(ids) for ids in inputs.input_ids]     
            
            # Generate with recording
            if test_run_time:
                start_time = time.time()
                    
            # Generate with sglang
            generated_results = model.generate(prompts, sampling_params=sampling_params)
            generated_texts = [result['text'] for result in generated_results]

            if test_run_time:
                end_time = time.time()
                run_time = end_time - start_time
            
        # Process results
        for j, (generated_text, item) in enumerate(zip(generated_texts, batch)):
            if '</think>' in generated_text:
                final_answer = generated_text.split('</think>')[1]
            else:
                final_answer = generated_text
            
            # Extract answer using the appropriate extractor
            predicted_answer, has_answer = answer_extractor(final_answer)
            
            # Check correctness
            is_correct = False
            if has_answer:
                is_correct = check_answer_correctness(predicted_answer, item['Answer'], answer_type)
            
            # Calculate token usage
            input_tokens = len(tokenizer.encode(prompts[j]))
            output_tokens = len(tokenizer.encode(generated_text))
            total_tokens = input_tokens + output_tokens
            
            if use_hybrid:
                recorder = recorders[j]
                stats = recorder.get_statistics()
                print(f"Total tokens: {stats['total_tokens']}")
                print(f"Quick model tokens: {stats['quick_model_tokens']} ({stats['quick_model_percentage']:.1f}%)")
                print(f"Reference model tokens: {stats['reference_model_tokens']} ({stats['reference_model_percentage']:.1f}%)")
                print(f"Overall model agreement: {stats['model_agreement_count']} tokens ({stats['model_agreement_percentage']:.1f}%)")
                print(f"Quick source agreement: {stats['quick_source_agreement_count']}/{stats['quick_source_total']} tokens ({stats['quick_source_agreement_percentage']:.1f}%)")
                print(f"Total parameters used: {stats['total_params_billions']:.1f}B")
                print(f"Average parameters per token: {stats['avg_params_billions']:.1f}B")
                
                # Extract key statistics
                quick_model_percentage = stats['quick_model_percentage']
                reference_model_percentage = stats['reference_model_percentage']
                model_agreement_percentage = stats['model_agreement_percentage']
                quick_source_agreement_percentage = stats['quick_source_agreement_percentage']
                total_params_billions = stats['total_params_billions']
                avg_params_billions = stats['avg_params_billions']
                
                result = {
                "problem_id": item['ID'],
                "correct_answer": item['Answer'],
                "has_extracted_answer": has_answer,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "full_output": generated_text,
                "quick_model_percentage": quick_model_percentage,
                "reference_model_percentage": reference_model_percentage,
                "model_agreement_percentage": model_agreement_percentage,
                "quick_source_agreement_percentage": quick_source_agreement_percentage,
                "total_params_billions": total_params_billions,
                "avg_params_billions": avg_params_billions,
                "run_time": run_time
            }
            else:
                result = {
                    "problem_id": item['ID'],
                    "correct_answer": item['Answer'],
                    "has_extracted_answer": has_answer,
                    "predicted_answer": predicted_answer,
                    "is_correct": is_correct,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "full_output": generated_text,
                    "run_time": run_time
                }
            
            # Add dataset-specific fields
            if dataset_config.get("answer_type") == "multiple_choice" and "Options" in item:
                result["options"] = item["Options"]
            if dataset_config.get("answer_type") == "mmlu-multiple-choice" and "Category" in item:
                result["category"] = item["Category"]
            results.append(result)
            
            # Save intermediate result to temp directory
            problem_id = item['ID']
            if problem_id in problem_id_counts:
                problem_id_counts[problem_id] += 1
                run_number = problem_id_counts[problem_id]
            else:
                problem_id_counts[problem_id] = 1
                run_number = 1
            
            temp_output_path = os.path.join(temp_dir, f"{problem_id}_run_{run_number}.txt")
            temp_output_csv_path = os.path.join(temp_csv_dir, f"{problem_id}_run_{run_number}.csv")
            write_to_file(temp_output_path, result)
            write_to_csv(temp_output_csv_path, result)
    
    if use_hybrid:
        generator.shutdown()
    else:
        model.shutdown()
    
    return results

def save_results(results: List[Dict], model_name: str, gpu_id: int, thread_id: int, output_dir: str):
    """Save evaluation results to CSV and full outputs to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save metrics to CSV
    df = pd.DataFrame([
        {k: v for k, v in result.items() if k != 'full_output'}
        for result in results
    ])
    csv_path = f"{output_dir}/results_{model_name}_gpu{gpu_id}_thread{thread_id}_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    # Save full outputs to separate text files
    outputs_dir = f"{output_dir}/outputs_{model_name}_gpu{gpu_id}_thread{thread_id}_{timestamp}"
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Track problem ID occurrences
    problem_id_counts = {}
    
    for result in results:
        problem_id = result['problem_id']
        
        # Update run number for this problem ID
        if problem_id in problem_id_counts:
            problem_id_counts[problem_id] += 1
            run_number = problem_id_counts[problem_id]
            output_path = f"{outputs_dir}/{problem_id}_run_{run_number}.txt"
        else:
            problem_id_counts[problem_id] = 1
            output_path = f"{outputs_dir}/{problem_id}_run_1.txt"
            
        write_to_file(output_path, result)
    
    # Save metadata to JSON
    json_path = f"{output_dir}/metadata_{model_name}_gpu{gpu_id}_thread{thread_id}_{timestamp}.json"
    metadata = {
        'model_name': model_name,
        'gpu_id': gpu_id,
        'thread_id': thread_id,
        'timestamp': timestamp,
        'num_problems': len(results),
        'outputs_dir': outputs_dir,
    }
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Results saved for GPU {gpu_id} thread {thread_id}")
    print(f"Full outputs saved in {outputs_dir}")

def combine_results(output_dir: str) -> Dict:
    """Combine all CSV results in the output directory, keeping latest version of duplicates.
    
    Returns:
        Dict: Dictionary containing average statistics
    """
    result_files = []
    for file in os.listdir(output_dir):
        if file.startswith("results_") and file.endswith(".csv"):
            # Extract timestamp from filename
            timestamp = file.split('_')[-1].replace('.csv', '')
            df = pd.read_csv(f"{output_dir}/{file}")
            df['timestamp'] = timestamp
            result_files.append(df)
    
    if not result_files:
        return {}
        
    # Combine all results
    combined_df = pd.concat(result_files, ignore_index=True)
    
    # Sort by timestamp (latest first) and keep first occurrence of each problem_id per model
    # combined_df = (combined_df
    #               .sort_values('timestamp', ascending=False)
    #               .groupby(['problem_id', 'model_name'])
    #               .first()
    #               .reset_index())
    combined_df = (combined_df
                  .sort_values('timestamp', ascending=False)
                  .reset_index())
    # Ensure token columns are included in the combined results
    # If any result file has these columns, they should be preserved
    essential_columns = ['input_tokens', 'output_tokens', 'total_tokens']
    
    # Check if any of the result files have these columns
    for col in essential_columns:
        if any(col in df.columns for df in result_files) and col not in combined_df.columns:
            print(f"Warning: Column '{col}' found in some result files but not in combined results. Adding with NaN values.")
            combined_df[col] = float('nan')
    
    # Save combined results
    combined_df.to_csv(f"{output_dir}/combined_results.csv", index=False)
    print(f"Combined {len(result_files)} result files, keeping latest version of duplicates.")
    
    # Calculate average statistics
    stats = {}
    
    # Calculate boolean column averages
    for col in ['has_extracted_answer', 'is_correct']:
        if col in combined_df.columns:
            stats[f'avg_{col}'] = float(combined_df[col].mean() * 100)  # Convert to percentage
    
    # Calculate percentage column averages
    for col in ['quick_model_percentage', 'reference_model_percentage', 
                'model_agreement_percentage', 'quick_source_agreement_percentage']:
        if col in combined_df.columns:
            stats[f'avg_{col}'] = float(combined_df[col].mean())
    
    # Calculate token usage averages
    for col in ['input_tokens', 'output_tokens', 'total_tokens']:
        if col in combined_df.columns:
            stats[f'avg_{col}'] = float(combined_df[col].mean())
    
    return stats

def get_completed_problems(output_dir: str) -> set:
    """Get set of problem IDs that have been successfully processed by reading temp txt files."""
    completed = set()
    temp_dir = os.path.join(output_dir, "temp")
    if not os.path.exists(temp_dir):
        return completed
    for fname in os.listdir(temp_dir):
        if fname.endswith('.txt'):
            # Extract problem ID by removing _run_$number$.txt suffix
            problem_id = fname.split('_run_')[0]
            completed.add(problem_id)
    return completed

def save_progress(output_dir: str, completed_problems: set):
    """Save progress tracking information."""
    progress_file = os.path.join(output_dir, 'progress.json')
    
    # Convert any numpy.int64 values to regular Python integers
    completed_problems_list = [int(x) if hasattr(x, 'item') else x for x in completed_problems]
    
    with open(progress_file, 'w') as f:
        json.dump({
            'completed_problems': completed_problems_list,
            'last_update': datetime.now().strftime("%Y%m%d_%H%M%S")
        }, f, indent=2)

def preprocess_dataset(dataset, dataset_config: Dict, save_result_dir: str) -> List[Dict]:
    """Preprocess dataset according to its configuration.
    
    Standardizes the dataset items to have consistent keys regardless of source dataset:
    - ID: unique problem identifier
    - Problem: formatted question text
    - Answer: correct answer in standardized format
    - Options: for multiple choice, a dictionary of options
    """
    processed_data = []
    
    # Get field mapping from config
    id_field = dataset_config.get("id_field", "ID")
    question_field = dataset_config.get("question_field", "Problem")
    answer_field = dataset_config.get("answer_field", "Answer")
    
    # Get filter configuration if it exists
    filter_config = dataset_config.get("filter", None)
    
    if dataset_config.get("answer_type") == "mmlu-multiple-choice":
        full_test_df = preprocess(dataset["test"])
        full_val_df = preprocess(dataset["validation"]) if "validation" in dataset else preprocess(dataset["test"])
        all_subjects = []
        for each in full_test_df:
            if each["category"] not in all_subjects:
                all_subjects.append(each["category"])
        if dataset_config.get("selected_subjects") == "all":
            selected_subjects = all_subjects
        else:
            selected_subjects = []
            args_selected = dataset_config.get("selected_subjects").split(",")
            for sub in all_subjects:
                for each in args_selected:
                    if each.replace(" ", "_") in sub.replace(" ", "_"):
                        selected_subjects.append(sub)
        print("selected subjects:\n" + "\n".join(selected_subjects))
        selected_subjects = sorted(selected_subjects)
        with open(os.path.join(save_result_dir, "summary.txt"), 'a') as f:
            f.write("\n------category level sta------\n")
        dataset = full_test_df
    
    if filter_config:
        dataset = dataset.filter(lambda x: x[filter_config["key"]] in filter_config["value"])
    
    for idx, item in enumerate(dataset):
        processed_item = {
            "ID": str(item[id_field]),
            "Problem": item[question_field],
            "Answer": item[answer_field]
        }
        
        # For multiple choice questions, process the options
        if dataset_config.get("answer_type") == "multiple_choice":
            options_fields = dataset_config.get("options_fields", [])
            if len(options_fields) >= 4:  # Need at least 4 options for A, B, C, D
                # Get the options in a consistent order
                options = [item[field] for field in options_fields]
                
                # Shuffle the options to randomize the correct answer position
                # Create a mapping from original positions to shuffled positions
                indices = list(range(len(options)))
                np.random.shuffle(indices)
                
                shuffled_options = [options[i] for i in indices]
                
                # Find where the correct answer ended up
                correct_index = indices.index(0)  # Assuming the first option is the correct one
                correct_letter = chr(65 + correct_index)  # A, B, C, D...
                
                processed_item["Options"] = {
                    "A": shuffled_options[0],
                    "B": shuffled_options[1],
                    "C": shuffled_options[2],
                    "D": shuffled_options[3]
                }
                processed_item["Answer"] = correct_letter
                
                # Format the problem with options
                processed_item["FormattedProblem"] = QUERY_TEMPLATE_MULTICHOICE.format(
                    Question=processed_item["Problem"],
                    A=processed_item["Options"]["A"],
                    B=processed_item["Options"]["B"],
                    C=processed_item["Options"]["C"],
                    D=processed_item["Options"]["D"]
                )
            else:
                print(f"Warning: Not enough option fields for multiple choice item {processed_item['ID']}")
                
        elif dataset_config.get("answer_type") == "mmlu-multiple-choice":
            k = 0
            prompt = generate_cot_prompt(full_val_df, item, k)
            processed_item["FormattedProblem"] = prompt
            processed_item["Answer"] = item["answer"]
            processed_item["Category"] = item["category"]
        
        elif dataset_config.get("answer_type") == "livecodebench":
            processed_item['__index'] = idx
            processed_item["__few_shots"] = False
            processed_item["prompt"], processed_item["inputs"], processed_item["outputs"] = lcb_codegeneration_prompt_fn(item)
            processed_item["FormattedProblem"] = processed_item["prompt"]
            processed_item['Answer'] = {
                'inputs': processed_item["inputs"],
                'outputs': processed_item["outputs"],
                "fn_name": json.loads(item["metadata"]).get("func_name", None),
            }
        
        else:
            # For non-multiple-choice, just use the prompt template if available
            template = dataset_config.get("prompt_template")
            if template:
                processed_item["FormattedProblem"] = template.format(question=processed_item["Problem"])
            else:
                processed_item["FormattedProblem"] = processed_item["Problem"]
                
        processed_data.append(processed_item)
        
    return processed_data

def extract_results_from_temp_csvs(output_dir: str,use_job_dirs: bool = True):
    """Extract and combine results from all temp CSV files in job directories.
    
    Args:
        output_dir: Directory containing job_* subdirectories with temp_csv folders
    """
    import glob
    import pandas as pd
    
    # Get all job directories
    if use_job_dirs:
        job_dirs = glob.glob(os.path.join(output_dir, "job_*"))
        if not job_dirs:
            print(f"No job directories found in {output_dir}")
            return
    else:
        job_dirs = [output_dir]
    
    # Collect all CSV files from temp_csv directories
    all_csv_files = []
    for job_dir in job_dirs:
        temp_csv_dir = os.path.join(job_dir, "temp_csv")
        print(f"temp_csv_dir: {temp_csv_dir}")
        if os.path.exists(temp_csv_dir):
            csv_files = glob.glob(os.path.join(temp_csv_dir, "*.csv"))
            all_csv_files.extend(csv_files)
    
    if not all_csv_files:
        print("No CSV files found in temp_csv directories")
        return
    
    print(f"Found {len(all_csv_files)} CSV files to process")
    
    # Read and combine all CSV files
    dfs = []
    for csv_file in all_csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    if not dfs:
        print("No valid CSV files could be read")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Sort by problem_id
    # combined_df = combined_df.sort_values('problem_id')
    
    # Save combined results
    output_path = os.path.join(output_dir, "combined_results.csv")
    combined_df.to_csv(output_path, index=False)
    print(f"Combined results saved to {output_path}")
    print(f"Total number of problems: {len(combined_df)}")

def print_evaluation_metrics(args: argparse.Namespace, output_dir: str, all_problems: List[Dict], completed_problems: set):
    """Print comprehensive evaluation metrics for all dataset types.
    
    Args:
        args: Command line arguments
        output_dir: Directory containing the results
        all_problems: List of all problems in the dataset
        completed_problems: Set of completed problem IDs
    """
    print("\n" + "="*80)
    print("EVALUATION METRICS SUMMARY")
    print("="*80)
    
    # Basic statistics
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model_path}")
    print(f"Total problems in dataset: {len(all_problems)}")
    
    # Load and analyze combined results if available
    combined_results_path = os.path.join(output_dir, "combined_results.csv")
    if os.path.exists(combined_results_path):
        try:
            df = pd.read_csv(combined_results_path)
            print(f"\nResults analyzed from: {len(df)} completed problems")
            
            # Basic accuracy metrics
            print("\n ACCURACY METRICS:")
            print("-" * 40)
            if 'has_extracted_answer' in df.columns:
                extraction_rate = df['has_extracted_answer'].mean() * 100
                print(f"Answer extraction rate: {extraction_rate:.2f}%")
            
            if 'is_correct' in df.columns:
                overall_accuracy = df['is_correct'].mean() * 100
                print(f"Overall accuracy: {overall_accuracy:.2f}%")
            
            # Pass@1 metrics (if repeat_input_num > 1)
            if args.repeat_input_num > 1 and 'is_correct' in df.columns:
                print(f"\n PASS@1 METRICS (repeat_input_num={args.repeat_input_num}):")
                print("-" * 40)
                
                # Calculate pass@1 as overall accuracy across all attempts
                overall_correct_count = df['is_correct'].sum()
                total_attempts = len(df)
                pass_at_1_rate = (overall_correct_count / total_attempts) * 100
                
                print(f"Pass@1: {pass_at_1_rate:.2f}% ({overall_correct_count}/{total_attempts})")
                
                # Additional statistics for repeated attempts
                grouped = df.groupby('problem_id')['is_correct']
                total_problems = len(grouped)
                attempts_per_problem = grouped.size()
                avg_attempts = attempts_per_problem.mean()
                print(f"Average attempts per problem: {avg_attempts:.1f}")
                print(f"Total problems: {total_problems}")
                
                # Show distribution of correct attempts per problem
                correct_per_problem = grouped.sum()
                print(f"Problems with 0 correct: {(correct_per_problem == 0).sum()}")
                print(f"Problems with all correct: {(correct_per_problem == args.repeat_input_num).sum()}")
                if args.repeat_input_num > 2:
                    print(f"Problems with partial correct: {((correct_per_problem > 0) & (correct_per_problem < args.repeat_input_num)).sum()}")
            
            # Token usage metrics
            print("\n TOKEN USAGE METRICS:")
            print("-" * 40)
            if 'input_tokens' in df.columns:
                avg_input_tokens = df['input_tokens'].mean()
                print(f"Average input tokens: {avg_input_tokens:.0f}")
            
            if 'output_tokens' in df.columns:
                avg_output_tokens = df['output_tokens'].mean()
                print(f"Average output tokens: {avg_output_tokens:.0f}")
            
            if 'total_tokens' in df.columns:
                avg_total_tokens = df['total_tokens'].mean()
                total_token_sum = df['total_tokens'].sum()
                print(f"Average total tokens: {avg_total_tokens:.0f}")
                print(f"Total tokens used: {total_token_sum:,}")
            
            # Performance metrics
            print("\n PERFORMANCE METRICS:")
            # Hybrid model metrics (if available)
            if 'quick_model_percentage' in df.columns:
                print("\n HYBRID MODEL METRICS:")
                print("-" * 40)
                avg_quick_percentage = df['quick_model_percentage'].mean()
                avg_ref_percentage = df['reference_model_percentage'].mean()
                print(f"Quick model usage: {avg_quick_percentage:.1f}%")
                print(f"Reference model usage: {avg_ref_percentage:.1f}%")
                
                if 'model_agreement_percentage' in df.columns:
                    avg_agreement = df['model_agreement_percentage'].mean()
                    print(f"Model agreement rate: {avg_agreement:.1f}%")
                
                if 'avg_params_billions' in df.columns:
                    avg_params = df['avg_params_billions'].mean()
                    print(f"Average parameters per token: {avg_params:.2f}B")
                
                # Calculate LLM output token ratio
                if 'output_tokens' in df.columns and 'input_tokens' in df.columns:
                    # Calculate LLM (reference model) generated tokens for each response
                    df['llm_generated_tokens'] = df['reference_model_percentage'] / 100.0 * (df['output_tokens'] - df['input_tokens'])
                    df['total_generated_tokens'] = df['output_tokens'] - df['input_tokens']
                    
                    # Calculate total LLM tokens and total generated tokens
                    total_llm_tokens = df['llm_generated_tokens'].sum()
                    total_generated_tokens = df['total_generated_tokens'].sum()
                    
                    # Calculate ratio
                    if total_generated_tokens > 0:
                        llm_token_ratio = (total_llm_tokens / total_generated_tokens) * 100
                        print(f"LLM output token ratio: {llm_token_ratio:.2f}%")
                        print(f"Total LLM generated tokens: {total_llm_tokens:.0f}")
                        print(f"Total generated tokens: {total_generated_tokens:.0f}")
                    else:
                        print("LLM output token ratio: N/A (no generated tokens)")
        
        except Exception as e:
            print(f"\nWarning: Could not analyze combined results: {e}")
    else:
        print(f"\nNote: Combined results file not found at {combined_results_path}")
    
    print("\n" + "="*80)
    print("END OF EVALUATION METRICS")
    print("="*80)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save args to JSON
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load and preprocess dataset
    print(f"Loading dataset: {args.dataset} from {args.dataset_path}")
    if args.dataset_config:
        print(f"Using dataset config: {args.dataset_config}")
        dataset = load_dataset(args.dataset_path, args.dataset_config, trust_remote_code=True)
    else:
        dataset = load_dataset(args.dataset_path, trust_remote_code=True)
    
    print(f"Preprocessing dataset as {args.dataset_config_dict['name']}")
        
    # Determine which dataset split to use based on config and type
    if args.dataset_config_dict.get("answer_type") == "mmlu-multiple-choice":
        dataset_split = dataset
    else:
        dataset_split = dataset['train'] if 'train' in dataset else dataset['test']
    
    all_problems = preprocess_dataset(dataset_split, args.dataset_config_dict, args.output_dir)
    
    print(f"Preprocessed {len(all_problems)} problems")
    
    # Determine which problems to process
    completed_problems = get_completed_problems(args.output_dir) if args.resume else set()
    
    if args.problem_ids:
        problems = [p for p in all_problems if p['ID'] in args.problem_ids.split(',')]
        print(f"Processing {len(problems)} specified problems")
    elif args.resume and not args.split_jobs:
        problems = [p for p in all_problems if p['ID'] not in completed_problems]
        print(f"Resuming with {len(problems)} remaining problems")
    else:
        problems = all_problems
    
    # Limit problem set for debug or testing
    if args.debug:
        problems = problems[:1]
        print("Debug mode: processing only first problem")
    elif args.num_problems:
        problems = problems[:args.num_problems]
        print(f"Processing first {args.num_problems} problems")
    
    if not problems:
        print("No problems to process!")
        if args.resume and not args.split_jobs:
            extract_results_from_temp_csvs(args.output_dir,use_job_dirs=False)
        else:
            extract_results_from_temp_csvs(args.output_dir)
        return
    
    print(f"Using data parallelism with tensor parallel size: {args.tp_size}")
    print(f"Output directory: {args.output_dir}")
    
    # Print example problem in debug mode
    # if args.debug and problems:
    #     print_example_problem(problems[0])
    
    # Process all problems using data parallelism
    if args.split_jobs:
        if args.job_id == -2:
            print("Job ID -2: extracting results for all temp csv files")
            extract_results_from_temp_csvs(args.output_dir)
            return
        if args.job_id == -1:
            # Only combine results, no processing
            print("Job ID -1: Combining results only, no processing")
            stats = combine_results(args.output_dir)
            
            # Get completed problems for metrics
            completed_problems = get_completed_problems(args.output_dir)
            
            # Print summary
            print("\nProcessing Summary:")
            print(f"Total problems in dataset: {len(all_problems)}")
            
            if stats:
                print(stats)
                stats_df = pd.DataFrame(stats, index=[0]).T
                stats_df.to_csv(f"{args.output_dir}/stats.csv", index=False)
            
            # Print comprehensive evaluation metrics
            print_evaluation_metrics(args, args.output_dir, all_problems, completed_problems)
            return
        else:
            # Process only a subset of problems for this job
            print(f"Job {args.job_id} of {args.job_nums}: Using automatic device mapping across all available GPUs")
            
            # Split problems evenly across jobs (same as multiprocessing)
            problems_per_job = math.ceil(len(problems) / args.job_nums)
            start_idx = args.job_id * problems_per_job
            end_idx = min((args.job_id + 1) * problems_per_job, len(problems))
            if start_idx >= len(problems):
                print(f"No problems to process for job {args.job_id}!")
                return
            job_problems = problems[start_idx:end_idx]
            print(f"Job {args.job_id} will process {len(job_problems)} problems (indices {start_idx} to {end_idx-1})") 
            
            if args.resume:
                completed_problems = get_completed_problems(args.output_dir)
                job_problems = [p for p in job_problems if p['ID'] not in completed_problems]
                print(f"Resuming with {len(job_problems)} remaining problems")
            
            # Process problems for this job
            results = process_problems(args, job_problems, args.use_hybrid)
            
            # Process results for this job
            newly_completed = get_completed_problems(args.output_dir)
            completed_problems.update(newly_completed)
            save_progress(args.output_dir, completed_problems)
            
            print(f"\nJob {args.job_id} Processing Summary:")
            print(f"Problems processed in this job: {len(job_problems)}")
            
            if len(completed_problems) < len(all_problems):
                remaining = set(p['ID'] for p in all_problems) - completed_problems
                print(f"Remaining problems: {len(remaining)}")
                
    else:
        results = process_problems(args, problems, args.use_hybrid)
        # Process results
        newly_completed = get_completed_problems(args.output_dir)
        completed_problems.update(newly_completed)
        save_progress(args.output_dir, completed_problems)
        stats = combine_results(args.output_dir)
        
        # Print summary
        print("\nProcessing Summary:")
        print(f"Total problems in dataset: {len(all_problems)}")
        print(f"Problems processed this run: {len(problems)}")
        
        if stats:
            print(stats)
            stats_df = pd.DataFrame(stats, index=[0]).T
            stats_df.to_csv(f"{args.output_dir}/stats.csv", index=False)
        
        if args.resume:
            # extract results from temp csvs
            extract_results_from_temp_csvs(args.output_dir,use_job_dirs=False)
            
    # Common code for suggesting how to process remaining problems
    if len(completed_problems) < len(all_problems):
        remaining = set(p['ID'] for p in all_problems) - completed_problems
        print(f"Remaining problems: {len(remaining)}")
        print("\nTo process remaining problems, run:")
        print(f"python evaluate/hf_dataset.py --dataset {args.dataset} --problem_ids '{','.join(sorted(remaining))}'")
        
    if args.dataset_config_dict.get("answer_type") == "mmlu-multiple-choice":
        print(f"python evaluate/mmlu_group_accuracy.py")
        df = pd.read_csv(f'{args.output_dir}/combined_results.csv')
        accuracy_by_category = df.groupby('category')['is_correct'].mean()
        accuracy_by_category_sorted = accuracy_by_category.sort_index()
        print("accuracy by category:")
        print("================")
        for category, accuracy in accuracy_by_category_sorted.items():
            print(f"{category}: {accuracy:.2%}")

        overall_accuracy = df['is_correct'].mean()
        print("\noverall accuracyll accuracy: {:.2%}".format(overall_accuracy))
    if args.dataset_config_dict.get("answer_type") == "livecodebench":
        print("evaluate livecodebench results")
        combined_results_path = f"{args.output_dir}/combined_results.csv"
        if os.path.exists(combined_results_path):
            os.system(f"python evaluate/livecodebench_answer_extractor.py --csv_path {combined_results_path}")
            print(f"livecodebench results saved in {args.output_dir}/combined_results_evaluation_light.csv")
        else:
            print(f"combined_results.csv not found in {args.output_dir}")

    # Print comprehensive evaluation metrics at the end
    if not args.split_jobs or args.job_id == -1:
        # Only print metrics for complete runs or when combining results
        print_evaluation_metrics(args, args.output_dir, all_problems, completed_problems)

    # if args.resume:
    #     # run resume_hf_sglang_results.py to get the results
    #     print(f"Running resume_hf_sglang_results.py to get the results")
    #     os.system(f"python evaluate/resume_hf_sglang_results.py --parent_dir {args.output_dir}")

def print_example_problem(example):
    """Print details of an example problem for debugging."""
    print("\nExample problem:")
    print(f"ID: {example['ID']}")
    print(f"Problem: {example['Problem']}")
    print(f"Formatted Problem: {example['FormattedProblem']}")
    print(f"Answer: {example['Answer']}")
    if "Options" in example:
        print("Options:")
        for key, value in example["Options"].items():
            print(f"  {key}: {value}")

def write_to_file(output_path: str, result: Dict):
    with open(output_path, 'w') as f:
        f.write(f"Problem ID: {result['problem_id']}\n")
        f.write(f"Correct Answer: {result['correct_answer']}\n")
        f.write(f"Has extracted answer: {result['has_extracted_answer']}\n")
        f.write(f"Predicted answer: {result['predicted_answer']}\n")
        f.write(f"Is correct: {result['is_correct']}\n")
        f.write(f"Input tokens: {result.get('input_tokens', 'N/A')}\n")
        f.write(f"Output tokens: {result.get('output_tokens', 'N/A')}\n")
        f.write(f"Total tokens: {result.get('total_tokens', 'N/A')}\n")
        f.write(f"Run time: {result.get('run_time', 'N/A')}\n")
        f.write("\nFull output:\n")
        f.write(result['full_output'])
            
        # Write additional statistics if available
        if 'quick_model_percentage' in result:
            f.write(f"\nQuick model percentage: {result['quick_model_percentage']}\n")
            f.write(f"Reference model percentage: {result['reference_model_percentage']}\n")
            f.write(f"Model agreement percentage: {result['model_agreement_percentage']}\n")
            f.write(f"Quick source agreement percentage: {result['quick_source_agreement_percentage']}\n")
            f.write(f"Total parameters billions: {result['total_params_billions']}\n")
            f.write(f"Average parameters billions: {result['avg_params_billions']}\n")
            
        # Write options if available (for multiple choice questions)
        if 'options' in result:
            f.write("\nOptions:\n")
            for option_key, option_text in result['options'].items():
                f.write(f"{option_key}: {option_text}\n")

def write_to_csv(output_path: str, result: Dict):
    """Write evaluation results to a CSV file.
    
    Args:
        output_path: Path where the CSV file will be saved
        result: Dictionary containing evaluation results
    """
    # Create a DataFrame with a single row
    df = pd.DataFrame([{
        'problem_id': result['problem_id'],
        'correct_answer': result['correct_answer'],
        'has_extracted_answer': result['has_extracted_answer'],
        'predicted_answer': result['predicted_answer'],
        'is_correct': result['is_correct'],
        'input_tokens': result.get('input_tokens', None),
        'output_tokens': result.get('output_tokens', None),
        'total_tokens': result.get('total_tokens', None),
        'run_time': result.get('run_time', None),
        'full_output': result['full_output']
    }])
    
    # Add hybrid model statistics if available
    if 'quick_model_percentage' in result:
        df['quick_model_percentage'] = result['quick_model_percentage']
        df['reference_model_percentage'] = result['reference_model_percentage']
        df['model_agreement_percentage'] = result['model_agreement_percentage']
        df['quick_source_agreement_percentage'] = result['quick_source_agreement_percentage']
        df['total_params_billions'] = result['total_params_billions']
        df['avg_params_billions'] = result['avg_params_billions']
    
    # Add options if available (for multiple choice questions)
    if 'options' in result:
        for option_key, option_text in result['options'].items():
            df[f'option_{option_key}'] = option_text
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    
if __name__ == "__main__":
    # Required for multiprocessing
    mp.set_start_method('spawn', force=True)
    main()
