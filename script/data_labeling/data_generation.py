import os
import sys
import subprocess
import time
import json
import glob
import argparse
from r2r.utils.config import MODEL_DICT

def parse_args():
    parser = argparse.ArgumentParser(
        description="Process inputs for training the router"
    )

    # SGLang configuration
    parser.add_argument(
        "--mem_fraction_static",
        type=float,
        default=0.7,
        help="Memory fraction for static allocation in SGLang",
    )
    parser.add_argument(
        "--tp_size", type=int, default=1, help="Tensor parallelism size for SGLang"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for SGLang batch processing",
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="Data type for model weights"
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the unified dataset created by step_-1_dataset_conversion.py or formatted dataset according to our requirements",
    )

    parser.add_argument(
        "--use_hf_dataset",
        action="store_true",
        help="Use HuggingFace dataset as input",
    )

    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Use HuggingFace/local dataset from train/test",
    )
    
    # Generation configuration
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32768,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature for generation"
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0, help="Top-p sampling parameter for generation"
    )
    parser.add_argument(
        "--top_k", type=int, default=-1, help="Top-k sampling parameter for generation"
    )

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Base directory to save results",
    )

    parser.add_argument(
        "--is_print",
        action="store_true",
        default=False,
        help="Print all model responses to standard output",
    )

    # Debug configuration
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (only process first item)",
    )
    parser.add_argument(
        "--num_items",
        type=int,
        default=None,
        help="Number of items to process (for testing)",
    )

    # Recovery configuration
    parser.add_argument(
        "--item_ids",
        type=str,
        default=None,
        help="Comma-separated list of specific item IDs to process",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint, processing only failed or missing items",
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default="script/data_labeling",
        help="Path to the data-generation programs",
    )
    parser.add_argument(
        "--skip_step_zero",
        action="store_true",
        default=False,
        help="Choose if skip the step 0:generating LLM response",
    )

    args = parser.parse_args()

    # Convert item IDs string to list if provided
    if args.item_ids:
        args.item_ids = [id.strip() for id in args.item_ids.split(",")]

    return args

def run_commands_sequentially(commands, continue_on_error=False):
    
    print(f"Start loading {len(commands)} programs...")
    
    for i, cmd in enumerate(commands, 1):
        print(f"\n Running Command ({i}/{len(commands)}): {cmd}")
        print("=" * 60)
        
        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1, 
                universal_newlines=True
            )
            
            
            for line in process.stdout:
                print(line, end='')
                
            
            process.wait()
            
            if process.returncode == 0:
                print(f" The command completed successfully (Exit code: {process.returncode})")
            else:
                print(f" Command failed while running the command : {cmd} (Exit code: {process.returncode})")
                if not continue_on_error:
                    print("The subsequent commands has been stopped")
                    return True
                    
        except Exception as e:
            print(f" Some Errors occured while running the command: {str(e)}")
            if not continue_on_error:
                print("The subsequent commands has been stopped")
                return True
    
    return False


def main():
    CONTINUE_ON_ERROR = False
    args = parse_args()
    output_dir=args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    program_path=args.program_path
    
    if not args.skip_step_zero:
        if args.use_hf_dataset:
            step_0_command = [
                f"python {program_path}/step_0_llm_response.py --output_dir {output_dir}/query_dataset_{args.split}/LLM_response --tp_size {args.tp_size} --dataset_path {args.dataset_path} --use_hf_dataset --split {args.split}"
            ]
        else:
            step_0_command = [
                f"python {program_path}/step_0_llm_response.py --output_dir {output_dir}/query_dataset_{args.split}/LLM_response --tp_size {args.tp_size} --dataset_path {args.dataset_path} --split {args.split}"
            ]
    
        fail_to_run = run_commands_sequentially(step_0_command, continue_on_error=CONTINUE_ON_ERROR)
        if fail_to_run:
            return

    step_1_command = [
        f"python {program_path}/step_1_slm_prefill.py --dataset_path {output_dir}/query_dataset_{args.split}/LLM_response/dataset_finished --output_path {output_dir}/query_dataset_{args.split}/LLM_response/SLM_prefill --top_p {args.top_p} --top_k {args.top_k} --temperature {args.temperature}"
    ]

    fail_to_run = run_commands_sequentially(step_1_command, continue_on_error=CONTINUE_ON_ERROR)

    if fail_to_run:
        return

    step_2_command = [
        f"python {program_path}/step_2_llm_continuation.py --input_path {output_dir}/query_dataset_{args.split}/LLM_response/SLM_prefill/prediction_comparison.csv --output_path {output_dir}/query_dataset_{args.split}/LLM_response/SLM_prefill/LLM_continuation_verify --tp_size {args.tp_size}"
    ]

    fail_to_run = run_commands_sequentially(step_2_command, continue_on_error=CONTINUE_ON_ERROR)

    if fail_to_run:
        return

    step_3_command = [
        f"python {program_path}/step_3_verify.py --input_csv {output_dir}/query_dataset_{args.split}/LLM_response/SLM_prefill/LLM_continuation_verify/generation_results_data_all_real_full.csv --output_csv {output_dir}/query_dataset_{args.split}/LLM_response/SLM_prefill/LLM_continuation_verify/generation_results_data_all_real_full_verify.csv --tp_size {args.tp_size}"
    ]

    fail_to_run = run_commands_sequentially(step_3_command, continue_on_error=CONTINUE_ON_ERROR)

    if fail_to_run:
        return

    step_4_command = [
        f"python {program_path}/step_4_construct_label_dataset.py --data_dir {output_dir}/query_dataset_{args.split}/LLM_response/SLM_prefill --csv LLM_continuation_verify/generation_results_data_all_real_full_verify.csv --output_sub_folder LLM_continuation_verify/divergent_label_dataset --divergent_column_name divergent"
    ]

    result = run_commands_sequentially(step_4_command, continue_on_error=CONTINUE_ON_ERROR)
    
    if result:
        return
    
    print("\n All commands have been completed!")
    return

if __name__ == "__main__":
    main()