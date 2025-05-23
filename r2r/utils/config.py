from dataclasses import dataclass
import torch

# ANSI color codes
QUICK_COLOR = "\033[34m"  # Dark blue
REFERENCE_COLOR = "\033[31m"  # Dark red
UNDERLINE = "\033[4m"  # Underline
RESET = "\033[0m"  # Reset all formatting

# Model configurations
MODEL_DICT = {
    "quick": 
        {
            "model_name": "DeepSeek-R1-Distill-Qwen-1.5B",
            "model_path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "param": "1.5",
            "mem_fraction_static": 0.15
        },
    "reference": 
        {
            "model_name": "DeepSeek-R1-Distill-Qwen-32B",
            "model_path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "param": "32",
            "mem_fraction_static": 0.80
        },
    "continuation_main":
        {
            "model_name": "DeepSeek-R1-Distill-Qwen-32B",
            "model_path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "param": "32",
            # "model_name": "DeepSeek-R1-Distill-Qwen-1.5B",
            # "model_path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            # "param": "1.5",
            "tp_size": 2,
            "base_gpu_id": 2,
            "mem_fraction_static": 0.90
        },
    "continuation_reference":
        {
            "model_name": "DeepSeek-R1-Distill-Qwen-32B",
            "model_path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "param": "32",
            "tp_size": 1,
            "base_gpu_id": 3,
            "mem_fraction_static": 0.90
        },
    "verify":
        {
            "model_name": "Qwen2.5-72B-Instruct",
            "model_path": "Qwen/Qwen2.5-72B-Instruct",
            "mem_fraction_static": 0.90,
            "base_gpu_id": 4,
            "tp_size": 4
        }
}
VOCABULARY_SIZE = 152064
TOTAL_GPU_NUM = 2

@dataclass
class TOKEN:
    MATCH = 0
    # mismatch includes both neutral and divergent
    NEUTRAL = 1
    DIVERGENT = 2