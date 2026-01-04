import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ.setdefault('MASTER_PORT', '29500')
import sys
import time
import argparse
import torch
import multiprocessing as mp
import sglang as sgl # Added for SGLang engine
from transformers import AutoTokenizer
import warnings

from r2r.models.dynamic_sglang_selector import DynamicSimpleSGLangSelector
from r2r.utils.config import (
    QUICK_COLOR, REFERENCE_COLOR, RESET, TOTAL_GPU_NUM,
    MODEL_DICT
)

# Suppress all warnings
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

def print_colored(text, color_segments):
    """
    Prints text with segments colored based on the source model.
    color_segments is a list of tuples: (segment_text, source_model)
    """
    output = []
    for segment_text, source_model in color_segments:
        if source_model == "quick":
            output.append(f"{QUICK_COLOR}{segment_text}{RESET}")
        elif source_model == "reference":
            output.append(f"{REFERENCE_COLOR}{segment_text}{RESET}")
        else: # Should not happen in this simplified version but good for robustness
            output.append(segment_text)
    print("".join(output), end='')
    sys.stdout.flush()


def run_simple_sglang_mode(args):
    print(f"Running in Native SGLang LLM")
    if not args.base_model_path:
        print("Error: Base model path is not configured. Please set reference model in config or provide --base_model_path.")
        return
    
    llm_engine = None
    try:
        llm_engine = sgl.Engine(
            model_path=args.base_model_path,
            tp_size=args.tp_size,
            skip_tokenizer_init=True,
            # Add other sgl.Engine parameters if needed, e.g., dtype="bfloat16"
            # Based on your other SGLang uses, you might want to specify dtype
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    except Exception as e:
        print(f"Failed to initialize SGLang Engine: {e}")
        if llm_engine: # Should not be necessary here, but good practice
            llm_engine.shutdown()
        return
    
    os.system('clear')
    print("SGLang Engine initialized. Enter your prompts (type 'exit' or 'quit' to end).")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting...")
                break
            if not user_input:
                continue
            
            current_turn_messages = [{"role": "user", "content": user_input}]
            # Use the SGLang engine's tokenizer to apply the chat template
            prompt_text = tokenizer.apply_chat_template(
                current_turn_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            input_tokens = [tokenizer.encode(prompt_text)]

            print("Bot: ")
            sys.stdout.flush()

            sampling_params_dict = {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
            }
            # Filter out None values, as SGLang might not accept them for all params
            sampling_params_dict = {k: v for k, v in sampling_params_dict.items() if v is not None}
            # Ensure temperature is explicitly set if it's 0.0 for greedy, SGLang should handle this.
            if args.temperature == 0.0 and sampling_params_dict.get("top_p", 1.0) >= 1.0:
                 # For greedy, sglang typically expects temperature = 0.0 or very close to it.
                 # top_p = 1.0 is default for greedy.
                 pass # Temperature is already set correctly

            full_generated_text = ""
            
            with PerformanceTimer() as timer:
                generated_stream = llm_engine.generate(
                    input_ids=input_tokens, 
                    sampling_params=sampling_params_dict,
                    stream=True
                )
                
                for output_item in generated_stream:
                    # print(output_item)
                    output_text = tokenizer.decode(output_item['output_ids'][-1:], skip_special_tokens=True)
                    print(f"{REFERENCE_COLOR}{output_text}{RESET}", end='')
                    sys.stdout.flush()
                    full_generated_text += output_text
            
            elapsed_time_s = timer.get_elapsed_time()
            
            print() # Newline after bot response is complete.

            num_output_tokens = len(tokenizer.encode(full_generated_text))
            tokens_per_second = num_output_tokens / elapsed_time_s if elapsed_time_s > 0 else 0 # Avoid division by zero
            
            print(f"Performance: Total output tokens: {num_output_tokens}, Time: {elapsed_time_s:.2f}s, Tokens/s: {tokens_per_second:.2f}")

        except KeyboardInterrupt:
            print("\nExiting...")
            if llm_engine:
                llm_engine.shutdown()
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            if llm_engine:
                llm_engine.shutdown()
            # Consider whether to break or continue on other exceptions
            break # For now, exit on error
    
    if llm_engine:
        llm_engine.shutdown()


def run_dynamic_sglang_mode(args):
    print(f"Running in Dynamic SGLang mode with classifier: {args.router_path}")
    print(f"Using TP size: {args.tp_size}")

    strategy_kwargs = {
        'model_path': args.router_path
    }
    if args.neural_threshold:
        strategy_kwargs['threshold'] = args.neural_threshold

    ref_model_path = args.reference_model_path if args.reference_model_path else MODEL_DICT['reference']['model_path']
    qck_model_path = args.quick_model_path if args.quick_model_path else MODEL_DICT['quick']['model_path']

    if not ref_model_path:
        print("Error: Reference model path is not configured for SGLang. Please set REFERENCE_MODEL_NAME_OR_PATH in config or provide --reference_model_path.")
        return
    if not qck_model_path:
        print("Error: Quick model path is not configured for SGLang. Please set QUICK_MODEL_NAME_OR_PATH in config or provide --quick_model_path.")
        return

    sglang_kwargs = {
        "dtype": "bfloat16",
        "tp_size": args.tp_size
    }

    generator = DynamicSimpleSGLangSelector(
        device="cuda", 
        dtype=torch.bfloat16,
        switching_strategy='neural',
        strategy_kwargs=strategy_kwargs,
        is_record=False, 
        sglang_kwargs=sglang_kwargs
    )

    os.system('clear')
    print("R2R generator initialized. Enter your prompts (type 'exit' or 'quit' to end).")
    print(f"{QUICK_COLOR}■ SLM output will be this color.{RESET}")
    print(f"{REFERENCE_COLOR}■ LLM output will be this color.{RESET}")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting...")
                break
            if not user_input:
                continue
            
            current_turn_messages = [{"role": "user", "content": user_input}]
            prompt_text = generator.tokenizer.apply_chat_template(
                current_turn_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            input_ids = [generator.tokenizer.encode(prompt_text)]

            print("Bot: ")
            sys.stdout.flush()

            with PerformanceTimer() as timer:
                generated_texts, recorders = generator.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    record_generation=True, 
                    print_tokens=True # Set to False if you don't want per-token printing from generator
                )
            
            elapsed_time_s = timer.get_elapsed_time()

            generated_text_full = generated_texts[0] 
            # The rest of the logic for printing colored output remains, 
            # but we need to ensure it doesn't interfere with timing or token count.
            # The per-token printing from the generator itself might be what you want for live output.
            # If you want to disable the generator's print_tokens and use your own loop for colored printing:
            # Change print_tokens=False above, and uncomment and adapt the loop below.

            # recorder = recorders[0]
            # if hasattr(recorder, 'records') and recorder.records:
            #     # ... (existing colored printing logic) ...
            # else:
            #     print(generated_text_full) # Fallback if no records
            # print() # Ensure newline if using custom print loop

            num_output_tokens = len(generator.tokenizer.encode(generated_text_full)) - len(input_ids[0])
            tokens_per_second = num_output_tokens / elapsed_time_s if elapsed_time_s > 0 else 0 # Avoid division by zero
            
            print(f"\nPerformance: Total output tokens: {num_output_tokens}, Time: {elapsed_time_s:.2f}s, Tokens/s: {tokens_per_second:.2f}")

        except KeyboardInterrupt:
            print("\nExiting...")
            if 'generator' in locals() and hasattr(generator, 'shutdown'):
                 generator.shutdown() 
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            if 'generator' in locals() and hasattr(generator, 'shutdown'):
                 generator.shutdown()

    if 'generator' in locals() and hasattr(generator, 'shutdown'):
        generator.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Interactive Chat with SGLang (Simple or Dynamic Mode)")
    parser.add_argument('--router_path', type=str, default=None,
                        help='Path to the critical classifier model. If None, runs in simple SGLang mode.')
    
    parser.add_argument('--base_model_path', type=str, default=None,
                        help='Path to the base LLM for simple SGLang mode. Defaults to reference model from config.py if not set.')
    parser.add_argument('--reference_model_path', type=str, default=None,
                        help='Path to the reference model for SGLang dynamic mode. Defaults to reference model from config.py if not set.')
    parser.add_argument('--quick_model_path', type=str, default=None,
                        help='Path to the quick model for SGLang dynamic mode. Defaults to quick model from config.py if not set.')

    parser.add_argument('--tp_size', type=int, default=TOTAL_GPU_NUM if TOTAL_GPU_NUM > 0 else 1,
                        help='Tensor parallelism size for SGLang. Defaults to TOTAL_GPU_NUM or 1.')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Temperature for sampling.')
    parser.add_argument('--top_p', type=float, default=1.0,
                        help='Top-p for nucleus sampling.')
    parser.add_argument('--max_new_tokens', type=int, default=32768,
                        help='Maximum new tokens to generate.')
    parser.add_argument('--neural_threshold', type=float, default=0.5,
                        help='Threshold for the neural switching strategy (default: 0.5).')

    args = parser.parse_args()

    if args.router_path:
        # This mode already uses SGLang via DynamicSimpleSGLangSelector
        run_dynamic_sglang_mode(args)
    else:
        if args.base_model_path is None:
            args.base_model_path = MODEL_DICT['reference']['model_path']
        run_simple_sglang_mode(args) # Changed here

if __name__ == "__main__":
    if torch.cuda.is_available():
        pass
    else:
        print("WARNING: CUDA not available. SGLang dynamic mode will likely fail.")

    mp.set_start_method("spawn", force=True)
    main()