"""
R2R Gradio Demo - Refactored Version with Pluggable Backends

This is the refactored version of the Gradio demo that uses the new
decoupled architecture with pluggable inference backends.

Usage:
    # Basic usage with R2R backend
    python gradio_demo_refactored.py --backend r2r
    
    # With custom router path
    python gradio_demo_refactored.py --backend r2r --router_path path/to/router.pt
    
    # Future: Multiple backends for comparison
    python gradio_demo_refactored.py --backend r2r --backend single_llm
"""

import os

# Set Gradio temp directory before importing gradio
GRADIO_TMP_DIR = "/share/geyi/R2R_extension/R2R/script/playground/tmp"
os.makedirs(GRADIO_TMP_DIR, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = GRADIO_TMP_DIR

# Clear proxy settings
for proxy_var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
    os.environ.pop(proxy_var, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

# R2R environment setup
os.environ['MASTER_ADDR'] = 'localhost'
os.environ.setdefault('MASTER_PORT', '29502')
os.environ["SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import sys
import argparse
import torch
import multiprocessing as mp
import warnings

warnings.filterwarnings("ignore")
torch.set_warn_always(False)

# Add project root to path for imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Import the new architecture components (relative to demo directory)
from script.playground.demo.backends import BackendConfig, BackendRegistry, get_default_registry
from script.playground.demo.ui import UIBuilder, create_demo_app


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="R2R Gradio Demo with Pluggable Backends"
    )
    
    # Backend selection
    parser.add_argument(
        '--backend',
        type=str,
        action='append',
        default=None,
        help='Backend(s) to use. Can be specified multiple times. Default: r2r'
    )
    
    # R2R specific options
    parser.add_argument(
        '--router_path',
        type=str,
        default='resource/default_router.pt',
        help='Path to the R2R router model'
    )
    parser.add_argument(
        '--neural_threshold',
        type=float,
        default=0.5,
        help='Threshold for neural routing strategy'
    )
    
    # Common options
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to the base model (for single_llm backend)'
    )
    parser.add_argument(
        '--tp_size',
        type=int,
        default=2,
        help='Tensor parallelism size'
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
        default=4096,
        help='Maximum new tokens to generate'
    )
    
    # Server options
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
        '--share',
        action='store_true',
        default=True,
        help='Create a shareable link'
    )
    
    # UI options
    parser.add_argument(
        '--logo_path',
        type=str,
        default='resource/logo.png',
        help='Path to logo image'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Default to r2r if no backends specified
    backends = args.backend or ['r2r']
    
    print("=" * 70)
    print("R2R Gradio Demo - Refactored Architecture")
    print("=" * 70)
    print(f"Backends: {', '.join(backends)}")
    print(f"Router Path: {args.router_path}")
    print(f"TP Size: {args.tp_size}")
    print(f"Threshold: {args.neural_threshold}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Max New Tokens: {args.max_new_tokens}")
    print("=" * 70)
    
    # Build configuration
    config = {
        'router_path': args.router_path,
        'neural_threshold': args.neural_threshold,
        'model_path': args.model_path,
        'tp_size': args.tp_size,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'max_new_tokens': args.max_new_tokens,
    }
    
    # Create UI builder
    builder = UIBuilder()
    
    # Add requested backends
    for backend_name in backends:
        print(f"\n[Main] Adding backend: {backend_name}")
        builder.add_backend(backend_name, **config)
    
    # Set logo if exists
    if os.path.exists(args.logo_path):
        builder.set_logo(args.logo_path)
    
    # Build the demo
    print("\n[Main] Building Gradio interface...")
    demo = builder.build(load_backends=True)
    
    print("\n" + "=" * 70)
    print("🚀 Launching Gradio interface...")
    print("=" * 70)
    
    try:
        # Use queue for proper CUDA context handling
        demo.queue(default_concurrency_limit=1)
        demo.launch(
            server_name=args.server_name,
            server_port=args.server_port,
            share=args.share,
        )
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        builder.shutdown()


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.device_count()} GPU(s)")
    else:
        print("WARNING: CUDA not available. R2R requires GPU support.")
    
    mp.set_start_method("spawn", force=True)
    main()
