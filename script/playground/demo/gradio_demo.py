"""
R2R Gradio Demo - Interactive Model Comparison

This creates a web interface to compare two inference modes:
1. Single LLM: Regular large language model inference
2. R2R: Dynamic routing between Small Language Model (SLM) and Large Language Model (LLM)

The R2R system intelligently routes tokens to the appropriate model based on 
reasoning divergence detection, achieving significant speedup while maintaining accuracy.

R2R runs in a separate subprocess for better isolation and stability.
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
os.environ.setdefault('MASTER_PORT', '29510')
os.environ["SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import sys
import time
import torch
import argparse
import gradio as gr
import multiprocessing as mp
from multiprocessing import Process, Queue
from typing import Optional, Generator, Tuple, List
import warnings

warnings.filterwarnings("ignore")
torch.set_warn_always(False)

# Import config (lightweight, no heavy CUDA dependencies)
from r2r.utils.config import (
    QUICK_COLOR, REFERENCE_COLOR, RESET, TOTAL_GPU_NUM,
    MODEL_DICT
)


# ============================================================================
# Shared Streaming Utilities
# ============================================================================

class StreamCapture:
    """
    Base class for capturing stdout and sending to queue in real-time.
    Used by both R2R and Single LLM subprocesses.
    
    Features throttled updates to prevent queue flooding when tokens are
    generated faster than the web UI can display them.
    """
    def __init__(self, queue, original_stdout, prefix="", update_interval=0.03):
        """
        Initialize stream capture with throttled updates.
        
        Args:
            queue: Multiprocessing queue to send updates to
            original_stdout: Original sys.stdout to write through
            prefix: Optional prefix for messages
            update_interval: Minimum seconds between queue updates (default 30ms)
        """
        import threading
        import re
        self.queue = queue
        self.original_stdout = original_stdout
        self.prefix = prefix
        self.update_interval = update_interval
        self.accumulated_text = ""
        self.accumulated_html = ""
        self.token_count = 0
        self.start_time = None
        self.last_update_time = 0
        self.lock = threading.Lock()
        self.capturing = False
        self.current_color = None
        self._pending_update = False
        # Pattern to match ANSI color codes
        self.ansi_pattern = re.compile(r'\033\[([0-9;]*)m')
        # Colors for HTML output
        self.SLM_COLOR = "#3B82F6"  # Blue
        self.LLM_COLOR = "#EF4444"  # Red
    
    def start(self):
        """Start capturing - reset state and record start time."""
        import time
        with self.lock:
            self.accumulated_text = ""
            self.accumulated_html = ""
            self.token_count = 0
            self.start_time = time.time()
            self.last_update_time = 0
            self.capturing = False
            self.current_color = None
            self._pending_update = False
    
    def _ansi_to_html_color(self, ansi_code):
        """Convert ANSI color code to HTML color."""
        if not ansi_code or ansi_code == '0':
            return None
        codes = ansi_code.split(';')
        for code in codes:
            if code in ('34', '94', '36', '96'):  # Blue/Cyan
                return self.SLM_COLOR
            elif code in ('31', '91', '35', '95'):  # Red/Magenta
                return self.LLM_COLOR
        return None
    
    def _escape_html(self, text):
        """Escape HTML special characters."""
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
    
    def _process_ansi_text(self, text):
        """Process text with ANSI codes and return (html, plain) tuple."""
        pos = 0
        result_html = ""
        plain_text = ""
        
        for match in self.ansi_pattern.finditer(text):
            # Add text before this ANSI code
            before_text = text[pos:match.start()]
            if before_text:
                escaped = self._escape_html(before_text)
                plain_text += before_text
                if self.current_color:
                    result_html += f'<span style="color: {self.current_color};">{escaped}</span>'
                else:
                    result_html += escaped
            
            # Update current color based on ANSI code
            ansi_code = match.group(1)
            new_color = self._ansi_to_html_color(ansi_code)
            if new_color is not None:
                self.current_color = new_color
            elif ansi_code == '0' or ansi_code == '':
                self.current_color = None
            
            pos = match.end()
        
        # Add remaining text after last ANSI code
        remaining = text[pos:]
        if remaining:
            escaped = self._escape_html(remaining)
            plain_text += remaining
            if self.current_color:
                result_html += f'<span style="color: {self.current_color};">{escaped}</span>'
            else:
                result_html += escaped
        
        return result_html, plain_text
    
    def _should_send_update(self):
        """Check if enough time has passed to send an update (throttling)."""
        import time
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self.last_update_time = current_time
            return True
        return False
    
    def write(self, text):
        """Capture writes to stdout with throttled queue updates."""
        import time
        # Always write to original stdout immediately
        self.original_stdout.write(text)
        self.original_stdout.flush()
        
        if '\033[' in text or self.capturing:
            self.capturing = True
            with self.lock:
                html_part, plain_part = self._process_ansi_text(text)
                
                if html_part or plain_part:
                    self.accumulated_html += html_part
                    self.accumulated_text += plain_part
                    self.token_count += 1
                    self._pending_update = True
                    
                    # Only send update if enough time has passed (throttling)
                    # This prevents queue flooding when tokens are generated very fast
                    if self._should_send_update():
                        elapsed = time.time() - self.start_time if self.start_time else 0
                        tokens_per_sec = self.token_count / elapsed if elapsed > 0 else 0
                        self._send_update(tokens_per_sec)
                        self._pending_update = False
    
    def flush_pending(self):
        """Force send any pending update (call before generation ends)."""
        import time
        with self.lock:
            if self._pending_update and self.accumulated_text:
                elapsed = time.time() - self.start_time if self.start_time else 0
                tokens_per_sec = self.token_count / elapsed if elapsed > 0 else 0
                self._send_update(tokens_per_sec)
                self._pending_update = False
    
    def _send_update(self, tokens_per_sec):
        """Override in subclass to send appropriate update message."""
        pass
    
    def flush(self):
        self.original_stdout.flush()
    
    def get_accumulated(self):
        with self.lock:
            return self.accumulated_html, self.accumulated_text, self.token_count


class R2RStreamCapture(StreamCapture):
    """Stream capture for R2R with colored HTML output and throttled updates."""
    
    def __init__(self, queue, original_stdout, prefix="", update_interval=0.03):
        super().__init__(queue, original_stdout, prefix, update_interval)
    
    def _send_update(self, tokens_per_sec):
        self.queue.put({
            "type": "generate_r2r_partial",
            "html_text": self.accumulated_html,
            "plain_text": self.accumulated_text,
            "stats": f"Generating... | Tokens: {self.token_count} | Speed: {tokens_per_sec:.1f} tok/s",
        })


class LLMStreamCapture(StreamCapture):
    """Stream capture for Single LLM with plain text output and throttled updates."""
    
    def __init__(self, queue, original_stdout, prefix="", update_interval=0.03):
        super().__init__(queue, original_stdout, prefix, update_interval)
    
    def _send_update(self, tokens_per_sec):
        self.queue.put({
            "type": "generate_partial",
            "text": self.accumulated_text,
            "stats": f"Generating... | Tokens: {self.token_count} | Speed: {tokens_per_sec:.1f} tok/s",
        })


# ============================================================================
# R2R Subprocess Worker
# ============================================================================

def r2r_subprocess_worker(
    request_queue: Queue,
    response_queue: Queue,
    router_path: str,
    base_model_path: Optional[str],
    tp_size: int,
    neural_threshold: float,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    gpu_ids: str = "0,1",
    stream_update_interval: float = 0.03,
):
    """
    Worker function that runs in a subprocess to handle R2R model loading and inference.
    
    This isolates the CUDA context and heavy model operations from the main Gradio process.
    
    Args:
        request_queue: Queue to receive requests from main process
        response_queue: Queue to send responses back to main process
        router_path: Path to the R2R router model
        base_model_path: Path to base LLM
        tp_size: Tensor parallelism size
        neural_threshold: Threshold for neural routing
        temperature: Sampling temperature
        top_p: Top-p sampling
        max_new_tokens: Max tokens to generate
        gpu_ids: Comma-separated GPU IDs to use (e.g., "0,1")
        stream_update_interval: Minimum interval between streaming updates (seconds)
    """
    # Set CUDA_VISIBLE_DEVICES for this subprocess to use specific GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    
    # Import heavy dependencies inside subprocess
    import sglang as sgl
    from transformers import AutoTokenizer
    from r2r.models.dynamic_sglang_selector import DynamicSimpleSGLangSelector
    from sglang.srt.sampling.sampling_params import SamplingParams
    
    print(f"[R2R Subprocess] Starting on GPUs: {gpu_ids}")
    
    # Get model paths from config
    reference_model_path = base_model_path or MODEL_DICT['reference']['model_path']
    quick_model_path = MODEL_DICT['quick']['model_path']
    reference_model_name = MODEL_DICT['reference']['model_name']
    quick_model_name = MODEL_DICT['quick']['model_name']
    
    # Load R2R model
    print(f"[R2R Subprocess] Loading R2R model...")
    print(f"  Router: {router_path}")
    print(f"  Quick Model: {quick_model_name}")
    print(f"  Reference Model: {reference_model_name}")
    print(f"  Threshold: {neural_threshold}")
    
    strategy_kwargs = {
        'model_path': router_path,
        'threshold': neural_threshold,
    }
    
    sglang_kwargs = {
        "dtype": "bfloat16",
        "tp_size": tp_size,
    }
    
    r2r_generator = DynamicSimpleSGLangSelector(
        device="cuda",
        dtype=torch.bfloat16,
        switching_strategy='neural',
        strategy_kwargs=strategy_kwargs,
        is_record=False,
        sglang_kwargs=sglang_kwargs,
    )
    
    print("[R2R Subprocess] ✓ Model loaded successfully!")
    
    # Signal that initialization is complete
    response_queue.put({"type": "init_complete", "status": "success"})
    
    # Main loop to handle requests
    while True:
        try:
            request = request_queue.get()
            
            if request is None or request.get("type") == "shutdown":
                print("[R2R Subprocess] Shutting down...")
                r2r_generator.shutdown()
                response_queue.put({"type": "shutdown_complete"})
                break
            
            if request.get("type") == "generate_r2r":
                user_input = request["user_input"]
                
                # Update threshold if provided in request (following pattern from interactive_chat.py)
                request_threshold = request.get("threshold")
                if request_threshold is not None:
                    # Update the threshold in the switching strategy
                    # The neural switching strategy has a threshold attribute that can be directly modified
                    try:
                        r2r_generator.switching_strategy.threshold = float(request_threshold)
                        print(f"[R2R Subprocess] Updated threshold to: {request_threshold}")
                    except AttributeError:
                        print(f"[R2R Subprocess] Warning: Could not update threshold - strategy may not support dynamic threshold")
                
                # Ensure CUDA context
                if torch.cuda.is_available():
                    if not torch.cuda.is_initialized():
                        torch.cuda.init()
                    torch.cuda.set_device(1)
                    _ = torch.zeros(1, device='cuda:1')
                    torch.cuda.synchronize(1)
                
                # Apply chat template
                messages = [{"role": "user", "content": user_input}]
                prompt_text = r2r_generator.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                input_ids = [r2r_generator.tokenizer.encode(prompt_text)]
                
                start_time = time.time()
                
                # Use shared R2R stream capture for real-time output with throttling
                original_stdout = sys.stdout
                capture = R2RStreamCapture(response_queue, original_stdout, update_interval=stream_update_interval)
                capture.start()
                sys.stdout = capture
                
                try:
                    # Generate with recording enabled - tokens will be captured in real-time
                    generated_texts, recorders = r2r_generator.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=-1,
                        record_generation=True,
                        print_tokens=True,
                    )
                finally:
                    # Flush any pending updates before restoring stdout
                    capture.flush_pending()
                    # Restore original stdout
                    sys.stdout = original_stdout
                
                elapsed_time = time.time() - start_time
                
                recorder = recorders[0] if recorders else None
                generated_text = generated_texts[0] if generated_texts else ""
                
                # Get the captured HTML from the stream capture
                captured_html, captured_plain, _ = capture.get_accumulated()
                
                # Calculate final statistics
                # Get current threshold for display
                current_threshold = neural_threshold
                if hasattr(r2r_generator, 'switching_strategy') and hasattr(r2r_generator.switching_strategy, 'threshold'):
                    current_threshold = r2r_generator.switching_strategy.threshold
                
                if recorder:
                    stats = recorder.get_statistics()
                    num_tokens = stats['total_tokens']
                    quick_pct = stats['quick_model_percentage']
                    ref_pct = stats['reference_model_percentage']
                    avg_params = stats['avg_params_billions']
                    tokens_per_sec = num_tokens / elapsed_time if elapsed_time > 0 else 0
                    
                    stats_text = (
                        f"✓ Complete | Tokens: {num_tokens} | Time: {elapsed_time:.2f}s | "
                        f"Speed: {tokens_per_sec:.1f} tok/s | Threshold: {current_threshold:.2f}\n"
                        f"SLM: {quick_pct:.1f}% | LLM: {ref_pct:.1f}% | "
                        f"Avg Params: {avg_params:.2f}B"
                    )
                else:
                    tokens_per_sec = len(r2r_generator.tokenizer.encode(generated_text)) / elapsed_time if elapsed_time > 0 else 0
                    stats_text = f"✓ Complete | Time: {elapsed_time:.2f}s | Speed: {tokens_per_sec:.1f} tok/s | Threshold: {current_threshold:.2f}"
                
                # Send final complete response with colored HTML
                response_queue.put({
                    "type": "generate_r2r_complete",
                    "html_text": captured_html if captured_html else generated_text,
                    "plain_text": generated_text,
                    "stats": stats_text,
                })
                
        except Exception as e:
            print(f"[R2R Subprocess] Error: {e}")
            import traceback
            traceback.print_exc()
            response_queue.put({
                "type": "error",
                "error": str(e),
            })


# ============================================================================
# Single LLM Subprocess Worker
# ============================================================================

def single_llm_subprocess_worker(
    request_queue: Queue,
    response_queue: Queue,
    model_path: str,
    tp_size: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    gpu_ids: str,
    stream_update_interval: float = 0.03,
):
    """
    Worker function that runs in a subprocess to handle Single LLM model loading and inference.
    
    This subprocess uses separate GPUs from the R2R subprocess for parallel comparison.
    Captures stdout in real-time for streaming output to web UI.
    
    Args:
        request_queue: Queue to receive requests from main process
        response_queue: Queue to send responses back to main process
        model_path: Path to the LLM model
        tp_size: Tensor parallelism size
        temperature: Sampling temperature
        top_p: Top-p sampling
        max_new_tokens: Max tokens to generate
        gpu_ids: Comma-separated GPU IDs to use (e.g., "2,3")
        stream_update_interval: Minimum interval between streaming updates (seconds)
    """
    # Set CUDA_VISIBLE_DEVICES for this subprocess to use specific GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    
    # Import heavy dependencies inside subprocess
    import sglang as sgl
    from transformers import AutoTokenizer
    from sglang.srt.sampling.sampling_params import SamplingParams
    
    print(f"[Single LLM Subprocess] Starting on GPUs: {gpu_ids}")
    
    # Get model path from config if not provided
    if model_path is None:
        model_path = MODEL_DICT['reference']['model_path']
    model_name = MODEL_DICT['reference']['model_name']
    
    print(f"[Single LLM Subprocess] Loading model...")
    print(f"  Model: {model_name}")
    print(f"  Path: {model_path}")
    print(f"  TP Size: {tp_size}")
    print(f"  GPUs: {gpu_ids}")
    
    # Load the model using SGLang Engine
    engine = sgl.Engine(
        model_path=model_path,
        tp_size=tp_size,
        dtype="bfloat16",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("[Single LLM Subprocess] ✓ Model loaded successfully!")
    
    # Signal that initialization is complete
    response_queue.put({"type": "init_complete", "status": "success"})
    
    # Main loop to handle requests
    while True:
        try:
            request = request_queue.get()
            
            if request is None or request.get("type") == "shutdown":
                print("[Single LLM Subprocess] Shutting down...")
                engine.shutdown()
                response_queue.put({"type": "shutdown_complete"})
                break
            
            if request.get("type") == "generate":
                user_input = request["user_input"]
                
                # Apply chat template
                messages = [{"role": "user", "content": user_input}]
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                start_time = time.time()
                generated_text = ""
                token_count = 0
                last_update_time = 0
                
                # Use SGLang's native streaming API for true real-time output
                sampling_params = {
                    "temperature": temperature, 
                    "top_p": top_p, 
                    "max_new_tokens": max_new_tokens
                }
                
                try:
                    # Stream generation - iterate over partial results
                    for chunk in engine.generate(
                        prompt=prompt_text,
                        sampling_params=sampling_params,
                        stream=True,  # Enable streaming mode
                    ):
                        # Extract text from streaming chunk
                        if hasattr(chunk, 'text'):
                            chunk_text = chunk.text
                        elif isinstance(chunk, dict) and 'text' in chunk:
                            chunk_text = chunk['text']
                        elif isinstance(chunk, str):
                            chunk_text = chunk
                        else:
                            chunk_text = str(chunk)
                        
                        # Update accumulated text (chunk.text contains full text so far)
                        generated_text = chunk_text
                        
                        # Count tokens in current output
                        current_tokens = len(tokenizer.encode(generated_text, add_special_tokens=False))
                        token_count = current_tokens
                        
                        # Throttle updates to prevent queue flooding
                        current_time = time.time()
                        if current_time - last_update_time >= stream_update_interval:
                            last_update_time = current_time
                            elapsed = current_time - start_time
                            tokens_per_sec = token_count / elapsed if elapsed > 0 else 0
                            
                            # Send streaming update to web UI
                            response_queue.put({
                                "type": "generate_partial",
                                "text": generated_text,
                                "stats": f"Streaming... | Tokens: {token_count} | Speed: {tokens_per_sec:.1f} tok/s",
                            })
                    
                    print()  # Newline after streaming completes
                    
                except Exception as e:
                    # Fallback to non-streaming if streaming fails
                    print(f"[Single LLM] Streaming failed, falling back to non-streaming: {e}")
                    result = engine.generate(
                        prompt=prompt_text,
                        sampling_params=sampling_params,
                    )
                    if hasattr(result, 'text'):
                        generated_text = result.text
                    elif isinstance(result, dict) and 'text' in result:
                        generated_text = result['text']
                    elif isinstance(result, str):
                        generated_text = result
                    else:
                        generated_text = str(result)
                    token_count = len(tokenizer.encode(generated_text, add_special_tokens=False))
                
                # Calculate final statistics
                elapsed_time = time.time() - start_time
                tokens_per_sec = token_count / elapsed_time if elapsed_time > 0 else 0
                
                stats_text = (
                    f"✓ Complete | Tokens: {token_count} | Time: {elapsed_time:.2f}s | "
                    f"Speed: {tokens_per_sec:.1f} tok/s"
                )
                
                # Send final complete response
                response_queue.put({
                    "type": "generate_complete",
                    "text": generated_text,
                    "stats": stats_text,
                })
                
        except Exception as e:
            print(f"[Single LLM Subprocess] Error: {e}")
            import traceback
            traceback.print_exc()
            response_queue.put({
                "type": "error",
                "error": str(e),
            })


# ============================================================================
# Single LLM Client (Main Process)
# ============================================================================

class SingleLLMSubprocessClient:
    """
    Client to communicate with the Single LLM subprocess from the main Gradio process.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        tp_size: int = 2,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: int = 4096,
        gpu_ids: str = "2,3",
        stream_update_interval: float = 0.03,
    ):
        """
        Initialize the Single LLM subprocess client.
        
        Starts a subprocess that loads and runs the LLM model on specified GPUs.
        
        Args:
            model_path: Path to the LLM model (defaults to reference model from config)
            tp_size: Tensor parallelism size
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_new_tokens: Maximum new tokens to generate
            gpu_ids: Comma-separated GPU IDs to use (e.g., "2,3")
            stream_update_interval: Minimum interval between streaming updates (seconds)
        """
        self.model_path = model_path or MODEL_DICT['reference']['model_path']
        self.model_name = MODEL_DICT['reference']['model_name']
        self.tp_size = tp_size
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.gpu_ids = gpu_ids
        self.stream_update_interval = stream_update_interval
        
        # Create communication queues
        self.request_queue = Queue()
        self.response_queue = Queue()
        
        print("=" * 70)
        print(f"Single LLM - Starting Subprocess on GPUs: {gpu_ids}")
        print("=" * 70)
        
        # Start the subprocess
        self.process = Process(
            target=single_llm_subprocess_worker,
            args=(
                self.request_queue,
                self.response_queue,
                model_path,
                tp_size,
                temperature,
                top_p,
                max_new_tokens,
                gpu_ids,
                stream_update_interval,
            ),
            daemon=False,
        )
        self.process.start()
        
        # Wait for initialization to complete
        print(f"[Main Process] Waiting for Single LLM subprocess to initialize...")
        response = self.response_queue.get(timeout=600)  # 10 minute timeout for model loading
        
        if response.get("type") == "init_complete" and response.get("status") == "success":
            print("[Main Process] ✓ Single LLM subprocess initialized successfully!")
        else:
            raise RuntimeError(f"Failed to initialize Single LLM subprocess: {response}")
        
        print("\n" + "=" * 70)
        print("✓ Single LLM Subprocess ready!")
        print("=" * 70 + "\n")
    
    def generate(self, user_input: str) -> Generator[Tuple[str, str], None, None]:
        """
        Generate response using single LLM.
        
        Yields (text, stats) tuples as tokens stream from subprocess.
        Uses non-blocking queue reads with short timeouts for responsive streaming.
        """
        from queue import Empty
        
        self.request_queue.put({
            "type": "generate",
            "user_input": user_input,
        })
        
        last_text = ""
        last_stats = ""
        start_time = time.time()
        max_wait_time = 600  # 10 minute overall timeout
        
        while True:
            try:
                # Use short timeout for responsive streaming
                response = self.response_queue.get(timeout=0.05)
                
                if response.get("type") == "error":
                    raise RuntimeError(f"Single LLM generation error: {response.get('error')}")
                
                if response.get("type") == "generate_partial":
                    text = response.get("text", "")
                    stats = response.get("stats", "")
                    # Only yield if content changed to reduce UI updates
                    if text != last_text or stats != last_stats:
                        last_text = text
                        last_stats = stats
                        yield text, stats
                
                elif response.get("type") == "generate_complete":
                    yield response.get("text", ""), response.get("stats", "")
                    break
                    
            except Empty:
                # Check for overall timeout
                if time.time() - start_time > max_wait_time:
                    raise TimeoutError("Single LLM generation timed out")
                # Continue waiting for more updates
                continue
    
    def shutdown(self):
        """Shutdown the Single LLM subprocess."""
        if self.process and self.process.is_alive():
            print("[Main Process] Sending shutdown signal to Single LLM subprocess...")
            self.request_queue.put({"type": "shutdown"})
            
            try:
                response = self.response_queue.get(timeout=30)
                if response.get("type") == "shutdown_complete":
                    print("[Main Process] ✓ Single LLM subprocess shut down gracefully")
            except:
                print("[Main Process] Forcing Single LLM subprocess termination...")
                self.process.terminate()
            
            self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.kill()


# ============================================================================
# R2R Client (Main Process)
# ============================================================================

class R2RSubprocessClient:
    """
    Client to communicate with the R2R subprocess from the main Gradio process.
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
        gpu_ids: str = "0,1",
        stream_update_interval: float = 0.03,
    ):
        """
        Initialize the R2R subprocess client.
        
        Starts a subprocess that loads and runs R2R models on specified GPUs.
        
        Args:
            router_path: Path to the R2R router model
            base_model_path: Path to base LLM (defaults to reference model from config)
            tp_size: Tensor parallelism size
            neural_threshold: Threshold for neural routing strategy
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_new_tokens: Maximum new tokens to generate
            gpu_ids: Comma-separated GPU IDs to use (e.g., "0,1")
            stream_update_interval: Minimum interval between streaming updates (seconds)
        """
        self.router_path = router_path
        self.tp_size = tp_size
        self.neural_threshold = neural_threshold
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.gpu_ids = gpu_ids
        self.stream_update_interval = stream_update_interval
        
        # Get model names for display
        self.reference_model_name = MODEL_DICT['reference']['model_name']
        self.quick_model_name = MODEL_DICT['quick']['model_name']
        
        # Create communication queues
        self.request_queue = Queue()
        self.response_queue = Queue()
        
        print("=" * 70)
        print(f"R2R Gradio Demo - Starting R2R Subprocess on GPUs: {gpu_ids}")
        print("=" * 70)
        
        # Start the subprocess
        self.process = Process(
            target=r2r_subprocess_worker,
            args=(
                self.request_queue,
                self.response_queue,
                router_path,
                base_model_path,
                tp_size,
                neural_threshold,
                temperature,
                top_p,
                max_new_tokens,
                gpu_ids,
                stream_update_interval,
            ),
            daemon=False,
        )
        self.process.start()
        
        # Wait for initialization to complete
        print("[Main Process] Waiting for R2R subprocess to initialize...")
        response = self.response_queue.get(timeout=600)  # 10 minute timeout for model loading
        
        if response.get("type") == "init_complete" and response.get("status") == "success":
            print("[Main Process] ✓ R2R subprocess initialized successfully!")
        else:
            raise RuntimeError(f"Failed to initialize R2R subprocess: {response}")
        
        print("\n" + "=" * 70)
        print("✓ R2R Subprocess ready!")
        print("=" * 70 + "\n")
    
    def generate_r2r(self, user_input: str, threshold: float = None) -> Generator[Tuple[str, str], None, None]:
        """
        Generate response using R2R dynamic routing with real-time streaming.
        
        Args:
            user_input: The user's input text
            threshold: Optional threshold override for this generation (0.0 to 1.0)
        
        Yields (html_text, stats) tuples as tokens stream from subprocess.
        HTML text includes color-coded spans for SLM (blue) and LLM (red) tokens.
        
        Uses non-blocking queue reads with short timeouts for responsive streaming.
        """
        from queue import Empty
        
        request = {
            "type": "generate_r2r",
            "user_input": user_input,
        }
        if threshold is not None:
            request["threshold"] = threshold
        
        self.request_queue.put(request)
        
        last_html = ""
        last_stats = ""
        start_time = time.time()
        max_wait_time = 600  # 10 minute overall timeout
        
        while True:
            try:
                # Use short timeout for responsive streaming
                response = self.response_queue.get(timeout=0.05)
                
                if response.get("type") == "error":
                    raise RuntimeError(f"R2R generation error: {response.get('error')}")
                
                if response.get("type") == "generate_r2r_partial":
                    # Return HTML text for colored display
                    html_text = response.get("html_text", response.get("plain_text", ""))
                    stats = response.get("stats", "")
                    # Only yield if content changed to reduce UI updates
                    if html_text != last_html or stats != last_stats:
                        last_html = html_text
                        last_stats = stats
                        yield html_text, stats
                
                elif response.get("type") == "generate_r2r_complete":
                    yield response.get("html_text", response.get("plain_text", "")), response.get("stats", "")
                    break
                    
            except Empty:
                if time.time() - start_time > max_wait_time:
                    raise TimeoutError("R2R generation timed out")
                continue
    
    def generate_single_llm(self, user_input: str) -> Generator[Tuple[str, str], None, None]:
        """
        Generate response using single LLM mode.
        
        Yields partial results as they stream from subprocess.
        Uses non-blocking queue reads with short timeouts for responsive streaming.
        """
        from queue import Empty
        
        self.request_queue.put({
            "type": "generate_single_llm",
            "user_input": user_input,
        })
        
        last_text = ""
        last_stats = ""
        start_time = time.time()
        max_wait_time = 600  # 10 minute overall timeout
        
        while True:
            try:
                # Use short timeout for responsive streaming
                response = self.response_queue.get(timeout=0.05)
                
                if response.get("type") == "error":
                    raise RuntimeError(f"Single LLM generation error: {response.get('error')}")
                
                if response.get("type") == "generate_single_llm_partial":
                    text = response.get("text", "")
                    stats = response.get("stats", "")
                    # Only yield if content changed to reduce UI updates
                    if text != last_text or stats != last_stats:
                        last_text = text
                        last_stats = stats
                        yield text, stats
                
                elif response.get("type") == "generate_single_llm_complete":
                    yield response.get("text", ""), response.get("stats", "")
                    break
                    
            except Empty:
                # Check for overall timeout
                if time.time() - start_time > max_wait_time:
                    raise TimeoutError("Single LLM generation timed out")
                # Continue waiting for more updates
                continue
    
    def shutdown(self):
        """Shutdown the R2R subprocess."""
        if self.process and self.process.is_alive():
            print("[Main Process] Sending shutdown signal to R2R subprocess...")
            self.request_queue.put({"type": "shutdown"})
            
            try:
                response = self.response_queue.get(timeout=30)
                if response.get("type") == "shutdown_complete":
                    print("[Main Process] ✓ R2R subprocess shut down gracefully")
            except:
                print("[Main Process] Forcing subprocess termination...")
                self.process.terminate()
            
            self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.kill()


# ============================================================================
# Gradio Demo UI
# ============================================================================

def create_demo(r2r_client: R2RSubprocessClient, single_llm_client: SingleLLMSubprocessClient) -> gr.Blocks:
    """
    Create Gradio interface for R2R demo with side-by-side comparison.
    
    Args:
        r2r_client: Initialized R2RSubprocessClient instance
        single_llm_client: Initialized SingleLLMSubprocessClient instance
        
    Returns:
        Gradio Blocks interface
    """
    
    # Preset example questions
    EXAMPLE_QUESTIONS = {
        "math": """Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.""",
        "reasoning": """A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?""",
        "coding": """Write a Python function to find the longest palindromic substring in a given string. Explain your approach and analyze the time complexity.""",
        "science": """Explain why the sky appears blue during the day but red/orange during sunset. Include the physics principles involved.""",
    }
    
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
        - 🔵 **SLM (Small):** {r2r_client.quick_model_name} ({MODEL_DICT['quick']['param']}B params)
        - 🔴 **LLM (Large):** {r2r_client.reference_model_name} ({MODEL_DICT['reference']['param']}B params)
        """)
        
        # R2R Settings section
        gr.Markdown("### ⚙️ R2R Settings")
        with gr.Row():
            threshold_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=r2r_client.neural_threshold,
                step=0.05,
                label="🧭 Router Threshold",
                info="Lower = more tokens routed to LLM (slower, more accurate). Higher = more tokens stay with SLM (faster, less accurate).",
            )
        
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
        
        # Three buttons: R2R, Single LLM, and Both (parallel)
        with gr.Row():
            submit_r2r_btn = gr.Button("🚀 Generate with R2R", variant="primary", scale=2)
            submit_llm_btn = gr.Button("🤖 Generate with LLM", variant="secondary", scale=2)
            submit_both_btn = gr.Button("⚡ Generate Both", variant="stop", scale=2)
            clear_btn = gr.Button("🗑️ Clear", scale=1)
        
        gr.Markdown("---")
        
        # Side-by-side output section
        gr.Markdown("## 📤 Output Comparison")
        
        with gr.Row():
            # R2R Output Column
            with gr.Column(scale=1):
                gr.Markdown("### 🛤️ R2R Output")
                
                # Color legend for R2R
                gr.HTML("""
                <div style="display: flex; gap: 15px; padding: 8px; background: #f1f5f9; border-radius: 6px; margin-bottom: 8px; font-size: 0.9em;">
                    <div style="display: flex; align-items: center; gap: 5px;">
                        <div style="width: 12px; height: 12px; border-radius: 3px; background-color: #3B82F6;"></div>
                        <span><strong>SLM</strong></span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 5px;">
                        <div style="width: 12px; height: 12px; border-radius: 3px; background-color: #EF4444;"></div>
                        <span><strong>LLM</strong></span>
                    </div>
                </div>
                """)
                
                # R2R output with colored text
                r2r_output = gr.HTML(
                    value="<div style='min-height: 300px; padding: 12px; border: 1px solid #e5e7eb; border-radius: 8px; background: #fafafa; font-family: inherit; white-space: pre-wrap; overflow-y: auto; max-height: 500px; font-size: 16px; line-height: 1.6;'></div>",
                )
                
                r2r_stats = gr.Textbox(
                    label="📊 R2R Statistics",
                    lines=2,
                    interactive=False,
                )
            
            # Single LLM Output Column
            with gr.Column(scale=1):
                gr.Markdown("### 🤖 Single LLM Output")
                
                # Info for Single LLM
                gr.HTML(f"""
                <div style="display: flex; gap: 15px; padding: 8px; background: #fef3c7; border-radius: 6px; margin-bottom: 8px; font-size: 0.9em;">
                    <span><strong>Model:</strong> {r2r_client.reference_model_name} ({MODEL_DICT['reference']['param']}B params)</span>
                </div>
                """)
                
                # Single LLM output
                llm_output = gr.HTML(
                    value="<div style='min-height: 300px; padding: 12px; border: 1px solid #e5e7eb; border-radius: 8px; background: #fffbeb; font-family: inherit; white-space: pre-wrap; overflow-y: auto; max-height: 500px; font-size: 16px; line-height: 1.6;'></div>",
                )
                
                llm_stats = gr.Textbox(
                    label="📊 LLM Statistics",
                    lines=2,
                    interactive=False,
                )
        
        gr.Markdown("---")
        
        # Helper functions
        def format_r2r_output(html_content):
            """Wrap R2R HTML content in a styled container."""
            base_style = "min-height: 300px; padding: 12px; border: 1px solid #e5e7eb; border-radius: 8px; background: #fafafa; font-family: inherit; white-space: pre-wrap; overflow-y: auto; max-height: 500px; font-size: 16px; line-height: 1.6;"
            if not html_content:
                return f"<div style='{base_style}'></div>"
            return f"<div style='{base_style}'>{html_content}</div>"
        
        def format_llm_output(text_content):
            """Wrap LLM text content in a styled container."""
            base_style = "min-height: 300px; padding: 12px; border: 1px solid #e5e7eb; border-radius: 8px; background: #fffbeb; font-family: inherit; white-space: pre-wrap; overflow-y: auto; max-height: 500px; font-size: 16px; line-height: 1.6;"
            if not text_content:
                return f"<div style='{base_style}'></div>"
            # Escape HTML and convert newlines
            escaped = text_content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            return f"<div style='{base_style}'>{escaped}</div>"
        
        def respond_r2r_formatted(user_input: str, threshold: float):
            """Handle R2R generation with formatted HTML output."""
            if not user_input.strip():
                yield format_r2r_output(""), ""
                return
            
            for html_text, stats in r2r_client.generate_r2r(user_input, threshold=threshold):
                yield format_r2r_output(html_text), stats
        
        def respond_llm_formatted(user_input: str):
            """Handle Single LLM generation with formatted output."""
            if not user_input.strip():
                yield format_llm_output(""), ""
                return
            
            for text, stats in single_llm_client.generate(user_input):
                yield format_llm_output(text), stats
        
        def respond_both_parallel(user_input: str, threshold: float):
            """
            Handle both R2R and Single LLM generation in parallel using threads.
            
            This function starts both generators in separate threads and yields
            combined updates to update both UI panels simultaneously.
            """
            import threading
            from queue import Queue, Empty
            
            if not user_input.strip():
                yield (format_r2r_output(""), "", format_llm_output(""), "")
                return
            
            # Shared state for both generators
            r2r_result = {"html": "", "stats": "", "done": False}
            llm_result = {"text": "", "stats": "", "done": False}
            update_queue = Queue()
            
            def run_r2r():
                """Thread function to run R2R generation."""
                try:
                    for html_text, stats in r2r_client.generate_r2r(user_input, threshold=threshold):
                        r2r_result["html"] = html_text
                        r2r_result["stats"] = stats
                        update_queue.put("r2r_update")
                except Exception as e:
                    r2r_result["stats"] = f"Error: {e}"
                finally:
                    r2r_result["done"] = True
                    update_queue.put("r2r_done")
            
            def run_llm():
                """Thread function to run Single LLM generation."""
                try:
                    for text, stats in single_llm_client.generate(user_input):
                        llm_result["text"] = text
                        llm_result["stats"] = stats
                        update_queue.put("llm_update")
                except Exception as e:
                    llm_result["stats"] = f"Error: {e}"
                finally:
                    llm_result["done"] = True
                    update_queue.put("llm_done")
            
            # Start both threads
            r2r_thread = threading.Thread(target=run_r2r, daemon=True)
            llm_thread = threading.Thread(target=run_llm, daemon=True)
            
            r2r_thread.start()
            llm_thread.start()
            
            # Yield initial state
            yield (
                format_r2r_output("Generating..."),
                "Starting R2R...",
                format_llm_output("Generating..."),
                "Starting LLM..."
            )
            
            # Poll for updates from both threads
            last_yield_time = time.time()
            min_yield_interval = 0.05  # 50ms minimum between yields
            
            while not (r2r_result["done"] and llm_result["done"]):
                try:
                    # Wait for any update with timeout
                    update_queue.get(timeout=0.1)
                    
                    # Throttle yields to prevent UI overload
                    current_time = time.time()
                    if current_time - last_yield_time >= min_yield_interval:
                        last_yield_time = current_time
                        yield (
                            format_r2r_output(r2r_result["html"]),
                            r2r_result["stats"],
                            format_llm_output(llm_result["text"]),
                            llm_result["stats"]
                        )
                except Empty:
                    # Timeout - check if both are done
                    continue
            
            # Drain any remaining updates in queue
            while not update_queue.empty():
                try:
                    update_queue.get_nowait()
                except Empty:
                    break
            
            # Wait for threads to finish
            r2r_thread.join(timeout=1.0)
            llm_thread.join(timeout=1.0)
            
            # Final yield with complete results
            yield (
                format_r2r_output(r2r_result["html"]),
                r2r_result["stats"],
                format_llm_output(llm_result["text"]),
                llm_result["stats"]
            )
        
        # Event handlers for R2R (with unique concurrency_id for parallel execution)
        submit_r2r_btn.click(
            fn=respond_r2r_formatted,
            inputs=[user_input, threshold_slider],
            outputs=[r2r_output, r2r_stats],
            concurrency_id="r2r_generation",
            concurrency_limit=1,
        )
        
        # Event handlers for Single LLM (with unique concurrency_id for parallel execution)
        submit_llm_btn.click(
            fn=respond_llm_formatted,
            inputs=[user_input],
            outputs=[llm_output, llm_stats],
            concurrency_id="llm_generation",
            concurrency_limit=1,
        )
        
        # Generate Both button - triggers both R2R and LLM truly in parallel using threads
        submit_both_btn.click(
            fn=respond_both_parallel,
            inputs=[user_input, threshold_slider],
            outputs=[r2r_output, r2r_stats, llm_output, llm_stats],
            concurrency_id="both_generation",
            concurrency_limit=1,
        )
        
        # Clear button clears all outputs
        clear_btn.click(
            fn=lambda: ("", format_r2r_output(""), "", format_llm_output(""), ""),
            inputs=None,
            outputs=[user_input, r2r_output, r2r_stats, llm_output, llm_stats],
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


# ============================================================================
# Main Entry Point
# ============================================================================

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
        default=8192,
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
    parser.add_argument(
        '--r2r_gpu_ids',
        type=str,
        default='0,1',
        help='Comma-separated GPU IDs for R2R subprocess (e.g., "0,1")'
    )
    parser.add_argument(
        '--single_llm_gpu_ids',
        type=str,
        default='2,3',
        help='Comma-separated GPU IDs for Single LLM subprocess (e.g., "2,3")'
    )
    parser.add_argument(
        '--stream_update_interval',
        type=float,
        default=0.015,
        help='Minimum interval (seconds) between streaming updates to web UI (default: 0.03s = 30ms)'
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
    print(f"R2R GPUs: {args.r2r_gpu_ids}")
    print(f"Single LLM GPUs: {args.single_llm_gpu_ids}")
    print(f"Stream Update Interval: {args.stream_update_interval}s ({int(1/args.stream_update_interval):.0f} updates/sec max)")
    print("=" * 70)
    
    # Initialize R2R subprocess client (this starts the R2R subprocess on GPU 0,1)
    r2r_client = R2RSubprocessClient(
        router_path=args.router_path,
        base_model_path=args.base_model_path,
        tp_size=args.tp_size,
        neural_threshold=args.neural_threshold,
        temperature=args.temperature,
        gpu_ids=args.r2r_gpu_ids,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        stream_update_interval=args.stream_update_interval,
    )
    
    # Initialize Single LLM subprocess client (on GPU 2,3)
    single_llm_client = SingleLLMSubprocessClient(
        model_path=args.base_model_path,
        tp_size=args.tp_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        gpu_ids=args.single_llm_gpu_ids,
        stream_update_interval=args.stream_update_interval,
    )
    
    # Create and launch demo with both clients
    demo = create_demo(r2r_client, single_llm_client)
    
    print("\n" + "=" * 70)
    print("🚀 Launching Gradio interface...")
    
    try:
        # Use queue with higher concurrency to allow R2R and LLM to run in parallel
        # Each has its own concurrency_id so they don't block each other
        demo.queue(default_concurrency_limit=2)
        demo.launch(
            share=True
        )
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        r2r_client.shutdown()
        single_llm_client.shutdown()
    finally:
        r2r_client.shutdown()
        single_llm_client.shutdown()


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.device_count()} GPU(s)")
    else:
        print("WARNING: CUDA not available. R2R requires GPU support.")
    
    mp.set_start_method("spawn", force=True)
    main()
