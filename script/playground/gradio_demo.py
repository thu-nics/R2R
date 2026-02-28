"""
R2R Gradio Demo - Interactive Model Comparison

This creates a web interface to compare two inference modes:
1. Single LLM: Regular large language model inference
2. R2R: Dynamic routing between Small Language Model (SLM) and Large Language Model (LLM)

The R2R system intelligently routes tokens to the appropriate model based on 
reasoning divergence detection, achieving significant speedup while maintaining accuracy.

"""

import os

# Set Gradio temp directory before importing gradio
GRADIO_TMP_DIR = "/share/geyi/R2R_extension/R2R/script/playground/tmp"
os.makedirs(GRADIO_TMP_DIR, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = GRADIO_TMP_DIR

# Clear proxy settings to allow Gradio to connect to localhost
# This must be done before importing gradio
for proxy_var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
    os.environ.pop(proxy_var, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

# R2R requires MASTER_ADDR for distributed communication
os.environ['MASTER_ADDR'] = 'localhost'
os.environ.setdefault('MASTER_PORT', '29502')
os.environ["SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import sys
import time
import torch
import argparse
import gradio as gr
import multiprocessing as mp
import sglang as sgl
from typing import Optional, Generator, Tuple, List
from transformers import AutoTokenizer
import warnings

warnings.filterwarnings("ignore")
torch.set_warn_always(False)

from r2r.models.dynamic_sglang_selector import DynamicSimpleSGLangSelector
from r2r.utils.config import (
    QUICK_COLOR, REFERENCE_COLOR, RESET, TOTAL_GPU_NUM,
    MODEL_DICT
)


class PerformanceTimer:
    """Timer for measuring generation performance."""
    
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


class R2RModelManager:
    """
    Manages loading and inference for R2R models.
    
    Supports two modes:
    1. Single LLM: Uses only the reference (large) model
    2. R2R: Dynamic routing between quick (small) and reference (large) models
    """
    
    def __init__(
        self,
        router_path: str,
        base_model_path: Optional[str] = None,
        tp_size: int = 2,
        neural_threshold: float = 0.5,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: int = 4096,
    ):
        """
        Initialize R2RModelManager.
        
        Args:
            router_path: Path to the R2R router model
            base_model_path: Path to the base LLM (defaults to reference model from config)
            tp_size: Tensor parallelism size
            neural_threshold: Threshold for neural routing strategy
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_new_tokens: Maximum new tokens to generate
        """
        self.router_path = router_path
        self.tp_size = tp_size
        self.neural_threshold = neural_threshold
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        
        # Get model paths from config
        self.reference_model_path = base_model_path or MODEL_DICT['reference']['model_path']
        self.quick_model_path = MODEL_DICT['quick']['model_path']
        self.reference_model_name = MODEL_DICT['reference']['model_name']
        self.quick_model_name = MODEL_DICT['quick']['model_name']
        
        # Model instances
        self.single_engine = None
        self.single_tokenizer = None
        self.r2r_generator = None
        
        print("=" * 70)
        print("R2R Gradio Demo - Initializing Models")
        print("=" * 70)
        
        self._load_all_models()
    
    def _load_single_model(self):
        """Load single LLM engine using SGLang."""
        print(f"\n[Single LLM] Loading {self.reference_model_name}...")
        print(f"  Path: {self.reference_model_path}")
        
        self.single_engine = sgl.Engine(
            model_path=self.reference_model_path,
            tp_size=self.tp_size,
            skip_tokenizer_init=True,
        )
        self.single_tokenizer = AutoTokenizer.from_pretrained(self.reference_model_path)
        print("[Single LLM] ✓ Model loaded")
    
    def _load_r2r_model(self):
        """Load R2R dynamic routing model."""
        print(f"\n[R2R] Loading dynamic routing model...")
        print(f"  Router: {self.router_path}")
        print(f"  Quick Model: {self.quick_model_name}")
        print(f"  Reference Model: {self.reference_model_name}")
        print(f"  Threshold: {self.neural_threshold}")
        
        strategy_kwargs = {
            'model_path': self.router_path,
            'threshold': self.neural_threshold,
        }
        
        sglang_kwargs = {
            "dtype": "bfloat16",
            "tp_size": self.tp_size,
        }
        
        self.r2r_generator = DynamicSimpleSGLangSelector(
            device="cuda",
            dtype=torch.bfloat16,
            switching_strategy='neural',
            strategy_kwargs=strategy_kwargs,
            is_record=False,
            sglang_kwargs=sglang_kwargs,
        )
        print("[R2R] ✓ Model loaded")
    
    def _load_all_models(self):
        """Load all models sequentially."""
        try:
            self._load_r2r_model()
            # Note: Single LLM mode shares the reference model from R2R
            # We don't need to load it separately
            print("\n" + "=" * 70)
            print("✓ All models loaded successfully!")
            print("=" * 70 + "\n")
        except Exception as e:
            print(f"\n✗ Error loading models: {e}")
            raise
    
    def generate_single_llm(self, user_input: str) -> Generator[Tuple[str, str], None, None]:
        """
        Generate response using single LLM mode (reference model only).
        
        Args:
            user_input: User's question/prompt
            
        Yields:
            Tuple of (generated_text, stats_text)
        """
        # Apply chat template
        messages = [{"role": "user", "content": user_input}]
        prompt_text = self.r2r_generator.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.r2r_generator.tokenizer.encode(prompt_text)
        
        generated_text = ""
        start_time = time.time()
        
        # Use R2R generator but force all tokens to reference model by setting threshold to 0
        # This effectively makes it behave like single LLM
        # For a true single model comparison, we use the extend_step method
        
        # Reset cache and initialize
        self.r2r_generator.reset_cache_simple()
        self.r2r_generator.reference_prefix_indices_list = [[]]
        
        from sglang.srt.sampling.sampling_params import SamplingParams
        
        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=-1,
            max_new_tokens=1,
            stop=[],
        )
        
        current_ids = list(input_ids)
        output_ids = []
        
        for step in range(self.max_new_tokens):
            # Get next token from reference model
            next_tokens = self.r2r_generator.extend_step(
                input_ids=[current_ids],
                input_indices=[0],
                sampling_params=sampling_params,
            )
            
            next_token = next_tokens[0]
            
            # Check for EOS
            if next_token in self.r2r_generator.tokenizer.all_special_ids:
                eos_ids = getattr(self.r2r_generator.tokenizer, 'eos_token_id', None)
                if eos_ids:
                    if isinstance(eos_ids, int):
                        eos_ids = [eos_ids]
                    if next_token in eos_ids:
                        break
            
            output_ids.append(next_token)
            current_ids.append(next_token)
            
            # Decode and yield
            generated_text = self.r2r_generator.tokenizer.decode(output_ids, skip_special_tokens=True)
            
            elapsed_time = time.time() - start_time
            tokens_per_sec = len(output_ids) / elapsed_time if elapsed_time > 0 else 0
            
            stats = f"Tokens: {len(output_ids)} | Time: {elapsed_time:.2f}s | Speed: {tokens_per_sec:.1f} tok/s"
            
            yield generated_text, stats
        
        # Final yield
        elapsed_time = time.time() - start_time
        tokens_per_sec = len(output_ids) / elapsed_time if elapsed_time > 0 else 0
        stats = f"✓ Complete | Tokens: {len(output_ids)} | Time: {elapsed_time:.2f}s | Speed: {tokens_per_sec:.1f} tok/s"
        yield generated_text, stats
    
    def generate_r2r(self, user_input: str) -> Tuple[str, str, str]:
        """
        Generate response using R2R dynamic routing.
        
        Args:
            user_input: User's question/prompt
            
        Returns:
            Tuple of (colored_html_text, plain_text, stats_text)
        """
        # Ensure CUDA context is properly set for the quick scheduler (GPU 1)
        # This is critical when called from Gradio's thread pool
        if torch.cuda.is_available():
            # Initialize CUDA if not already done in this thread
            if not torch.cuda.is_initialized():
                torch.cuda.init()
            # Quick model runs on GPU 1, reference model uses GPU 0,1
            torch.cuda.set_device(1)
            # Create a small tensor to ensure CUDA context is active
            _ = torch.zeros(1, device='cuda:1')
            torch.cuda.synchronize(1)
        
        # Apply chat template
        messages = [{"role": "user", "content": user_input}]
        prompt_text = self.r2r_generator.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = [self.r2r_generator.tokenizer.encode(prompt_text)]
        
        start_time = time.time()
        
        # Generate with recording enabled
        generated_texts, recorders = self.r2r_generator.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=-1,
            record_generation=True,
            print_tokens=True,  # Print tokens in terminal for debugging
        )
        
        elapsed_time = time.time() - start_time
        
        # Get the recorder for statistics
        recorder = recorders[0] if recorders else None
        generated_text = generated_texts[0] if generated_texts else ""
        
        # Build colored HTML output
        colored_html = self._build_colored_html(recorder)
        
        # Calculate statistics
        if recorder:
            stats = recorder.get_statistics()
            num_tokens = stats['total_tokens']
            quick_pct = stats['quick_model_percentage']
            ref_pct = stats['reference_model_percentage']
            avg_params = stats['avg_params_billions']
            tokens_per_sec = num_tokens / elapsed_time if elapsed_time > 0 else 0
            
            stats_text = (
                f"✓ Complete | Tokens: {num_tokens} | Time: {elapsed_time:.2f}s | "
                f"Speed: {tokens_per_sec:.1f} tok/s\n"
                f"SLM: {quick_pct:.1f}% | LLM: {ref_pct:.1f}% | "
                f"Avg Params: {avg_params:.2f}B"
            )
        else:
            tokens_per_sec = len(self.r2r_generator.tokenizer.encode(generated_text)) / elapsed_time if elapsed_time > 0 else 0
            stats_text = f"✓ Complete | Time: {elapsed_time:.2f}s | Speed: {tokens_per_sec:.1f} tok/s"
        
        return colored_html, generated_text, stats_text
    
    def _build_colored_html(self, recorder) -> str:
        """
        Build HTML with colored text based on which model generated each token.
        
        Args:
            recorder: GenerationRecorder with token records
            
        Returns:
            HTML string with colored spans
        """
        if not recorder or not recorder.records:
            return ""
        
        # Define colors (matching the terminal colors but for HTML)
        slm_color = "#3B82F6"  # Blue for quick/SLM
        llm_color = "#EF4444"  # Red for reference/LLM
        
        html_parts = []
        
        for record in recorder.records:
            token_str = record.token_str
            # Escape HTML special characters
            token_str = token_str.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            # Handle newlines
            token_str = token_str.replace("\n", "<br>")
            
            if record.source_model == "quick":
                color = slm_color
                title = f"SLM ({self.quick_model_name})"
            else:
                color = llm_color
                title = f"LLM ({self.reference_model_name})"
            
            html_parts.append(f'<span style="color: {color};" title="{title}">{token_str}</span>')
        
        return "".join(html_parts)
    
    def shutdown(self):
        """Shutdown all model engines."""
        if self.r2r_generator:
            self.r2r_generator.shutdown()
        if self.single_engine:
            self.single_engine.shutdown()


def create_demo(model_manager: R2RModelManager) -> gr.Blocks:
    """
    Create Gradio interface for R2R demo.
    
    Args:
        model_manager: Initialized R2RModelManager instance
        
    Returns:
        Gradio Blocks interface
    """
    
    # Preset example questions
    EXAMPLE_QUESTIONS = {
        "math": """Find all positive integers n such that n² + 1 divides n³ + n² + n + 1.""",
        "reasoning": """A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?""",
        "coding": """Write a Python function to find the longest palindromic substring in a given string. Explain your approach and analyze the time complexity.""",
        "science": """Explain why the sky appears blue during the day but red/orange during sunset. Include the physics principles involved.""",
    }
    
    def respond_single(user_input: str):
        """Handle single LLM generation."""
        if not user_input.strip():
            yield "", ""
            return
        
        for text, stats in model_manager.generate_single_llm(user_input):
            yield text, stats
    
    def respond_r2r(user_input: str):
        """Handle R2R generation."""
        if not user_input.strip():
            return "", "", ""
        
        return model_manager.generate_r2r(user_input)
    
    def respond_both(user_input: str):
        """Handle generation for both modes (for comparison)."""
        if not user_input.strip():
            return "", "", ""
        
        return model_manager.generate_r2r(user_input)
    
    # Create Gradio interface
    with gr.Blocks(title="R2R Demo - Roads to Rome") as demo:
        
        # Header
        with gr.Row():
            with gr.Column(scale=1, min_width=100):
                gr.Image(
                    "resource/logo.png",
                    show_label=False,
                )
            with gr.Column(scale=5):
                gr.Markdown("""
                # 🛤️ Roads to Rome (R2R) Demo
                ### Efficiently Navigating Divergent Reasoning Paths with Small-Large Model Token Routing
                
                R2R intelligently routes **only critical, reasoning-divergent tokens to the large model**, 
                achieving significant speedup while maintaining accuracy.
                """)
        
        gr.Markdown("---")
        
        # Model info
        gr.Markdown(f"""
        **Models:**
        - 🔵 **SLM (Small):** {model_manager.quick_model_name} ({MODEL_DICT['quick']['param']}B params)
        - 🔴 **LLM (Large):** {model_manager.reference_model_name} ({MODEL_DICT['reference']['param']}B params)
        - 🧭 **Router Threshold:** {model_manager.neural_threshold}
        """)
        
        gr.Markdown("---")
        
        # Input section
        gr.Markdown("## 💬 Input")
        
        gr.Markdown("**Example Questions:**")
        with gr.Row():
            math_btn = gr.Button("🔢 Math Problem", size="sm")
            reasoning_btn = gr.Button("🧠 Reasoning", size="sm")
            coding_btn = gr.Button("💻 Coding", size="sm")
            science_btn = gr.Button("🔬 Science", size="sm")
        
        user_input = gr.Textbox(
            label="Your Question",
            placeholder="Type your question here or select an example above...",
            lines=3,
            max_lines=10,
        )
        
        with gr.Row():
            submit_btn = gr.Button("🚀 Generate with R2R", variant="primary", scale=2)
            clear_btn = gr.Button("🗑️ Clear", scale=1)
        
        gr.Markdown("---")
        
        # Output section
        gr.Markdown("## 📤 R2R Output")
        
        # Legend for colors
        gr.HTML("""
        <div style="display: flex; gap: 20px; padding: 10px; background: #f1f5f9; border-radius: 8px; margin-bottom: 10px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 16px; height: 16px; border-radius: 4px; background-color: #3B82F6;"></div>
                <span><strong>Blue:</strong> SLM (Small Language Model)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 16px; height: 16px; border-radius: 4px; background-color: #EF4444;"></div>
                <span><strong>Red:</strong> LLM (Large Language Model)</span>
            </div>
        </div>
        """)
        
        # R2R output with colored text
        r2r_colored_output = gr.HTML(
            label="R2R Response (Colored by Model)",
        )
        
        # Plain text output (hidden but useful for copying)
        r2r_plain_output = gr.Textbox(
            label="Plain Text Response",
            lines=10,
            max_lines=30,
            interactive=False,
            visible=True,
        )
        
        # Statistics
        r2r_stats = gr.Textbox(
            label="📊 Performance Statistics",
            lines=2,
            interactive=False,
        )
        
        gr.Markdown("---")
        
        # Event handlers
        submit_btn.click(
            fn=respond_r2r,
            inputs=[user_input],
            outputs=[r2r_colored_output, r2r_plain_output, r2r_stats],
        )
        
        user_input.submit(
            fn=respond_r2r,
            inputs=[user_input],
            outputs=[r2r_colored_output, r2r_plain_output, r2r_stats],
        )
        
        clear_btn.click(
            fn=lambda: ("", "", "", ""),
            inputs=None,
            outputs=[user_input, r2r_colored_output, r2r_plain_output, r2r_stats],
        )
        
        # Example button handlers
        math_btn.click(
            fn=lambda: EXAMPLE_QUESTIONS["math"],
            inputs=None,
            outputs=[user_input],
        )
        
        reasoning_btn.click(
            fn=lambda: EXAMPLE_QUESTIONS["reasoning"],
            inputs=None,
            outputs=[user_input],
        )
        
        coding_btn.click(
            fn=lambda: EXAMPLE_QUESTIONS["coding"],
            inputs=None,
            outputs=[user_input],
        )
        
        science_btn.click(
            fn=lambda: EXAMPLE_QUESTIONS["science"],
            inputs=None,
            outputs=[user_input],
        )
        
        # Disclaimer
        gr.Markdown("---")
        gr.Markdown("""
        ### ⚠️ Disclaimer
        
        This demo is provided for **research purposes only** on an **"AS-IS" basis without warranties of any kind**.
        
        - R2R models are in experimental stages and may produce inaccurate or inconsistent outputs.
        - Generated outputs do not represent the views or opinions of the creators.
        - **Users are solely responsible** for any use of generated content.
        
        ---
        
        **R2R** combines small and large language models by routing only critical tokens.
        [📑 Paper](https://arxiv.org/abs/2505.21600) | [🌐 Project Page](https://fuvty.github.io/R2R_Project_Page/) | [🤗 HuggingFace](https://huggingface.co/papers/2505.21600)
        
        **Feel free to star our repo or cite our paper if you find it useful!** ⭐
        """)
    
    return demo


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="R2R Gradio Demo")
    
    parser.add_argument(
        '--router_path',
        type=str,
        default='resource/default_router.pt',
        help='Path to the R2R router model'
    )
    parser.add_argument(
        '--base_model_path',
        type=str,
        default=None,
        help='Path to the base LLM (defaults to reference model from config)'
    )
    parser.add_argument(
        '--tp_size',
        type=int,
        default=TOTAL_GPU_NUM if TOTAL_GPU_NUM > 0 else 2,
        help='Tensor parallelism size'
    )
    parser.add_argument(
        '--neural_threshold',
        type=float,
        default=0.5,
        help='Threshold for neural routing strategy'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=1.0,
        help='Top-p sampling parameter'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=2048,
        help='Maximum new tokens to generate'
    )
    parser.add_argument(
        '--server_name',
        type=str,
        default='0.0.0.0',
        help='Server hostname'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=7860,
        help='Server port'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("R2R Gradio Demo")
    print("=" * 70)
    print(f"Router Path: {args.router_path}")
    print(f"TP Size: {args.tp_size}")
    print(f"Threshold: {args.neural_threshold}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Max New Tokens: {args.max_new_tokens}")
    print("=" * 70)
    
    # Initialize model manager
    model_manager = R2RModelManager(
        router_path=args.router_path,
        base_model_path=args.base_model_path,
        tp_size=args.tp_size,
        neural_threshold=args.neural_threshold,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )
    
    # Create and launch demo
    demo = create_demo(model_manager)
    
    print("\n" + "=" * 70)
    print("🚀 Launching Gradio interface...")
    
    try:
        # Use queue to ensure proper CUDA context handling
        demo.queue(default_concurrency_limit=1)
        demo.launch(
            share=True
        )
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        model_manager.shutdown()
    finally:
        model_manager.shutdown()


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.device_count()} GPU(s)")
    else:
        print("WARNING: CUDA not available. R2R requires GPU support.")
    
    mp.set_start_method("spawn", force=True)
    main()
