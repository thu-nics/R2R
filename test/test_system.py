import os

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29502"
import sys
import time
import argparse
import torch
import multiprocessing as mp
import sglang as sgl  # Added for SGLang engine
from transformers import AutoTokenizer
import warnings

from r2r.models.batch_inference.sl_disaggregation_system import SLDisaggregationSystem
from r2r.utils.config import QUICK_COLOR, REFERENCE_COLOR, RESET, TOTAL_GPU_NUM, MODEL_DICT


class PerformanceTimer:
    def __init__(self):
        self.start_event = None
        self.end_event = None
        self.start_time_cpu = None
        self.elapsed_time_s = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            print(f"***")
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


# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ["SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
torch.set_warn_always(False)


def main():
    qck_model_path = MODEL_DICT["quick"]["model_path"]

    quick_sglang_kwargs = {"dtype": "bfloat16", "tp_size": 1, "enable_return_hidden_states": True}
    reference_sglang_kwargs = {"dtype": "bfloat16", "tp_size": 1}
    strategy_kwargs = {"model_path": "R2R_router/my_router.pt"}
    strategy_kwargs["threshold"] = 0.416

    generator = SLDisaggregationSystem(
        device="cuda",
        dtype=torch.bfloat16,
        switching_strategy="neural",
        strategy_kwargs=strategy_kwargs,
        is_record=False,
        quick_sglang_kwargs=quick_sglang_kwargs,
        reference_sglang_kwargs=reference_sglang_kwargs,
        overlap_tp_schedule=False, # Default False
    )

    print(f"Ready to generate with SL Disaggreragion System.")
    batch_input_ids = []
    try:
        input_file = os.path.join(os.path.dirname(__file__), "input_text.txt")  # 即 test/slserver/input_text.txt
        if not os.path.isfile(input_file):
            raise FileNotFoundError(f"未找到输入文件: {input_file}")
        with open(input_file, "r", encoding="utf-8") as f:
            raw_lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        # if len(raw_lines) != 8:
        #    print(f"警告: 期望 8 行，实际 {len(raw_lines)} 行。仍继续处理。")
        for line in raw_lines:
            messages = [{"role": "user", "content": line}]
            prompt_text = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            print()
            token_ids = generator.tokenizer.encode(prompt_text)
            batch_input_ids.append(token_ids)
        # 可选：打印汇总信息
        print(f"已准备批量输入，共 {len(batch_input_ids)} 条，示例首条 token 数: {len(batch_input_ids[0]) if batch_input_ids else 0}")
    except Exception as e:
        print(f"批量输入准备失败: {e}")
        batch_input_ids = []

    # TODO: test the speed

    """
    start_time = time.time()
    result = generator.generate(batch_input_ids, max_new_tokens=8192, temperature=0, top_p=1, top_k=-1)
    end_time = time.time()
    """

    leng = 1
    start_time = time.time()
    result = generator.generate(batch_input_ids[:leng], max_new_tokens=2048, temperature=0, top_p=1, top_k=-1, display_progress=True)
    end_time = time.time()
    total_tokens = 0
    for obj in result:
        if isinstance(obj, dict):
            total_tokens += len(obj.get("output_ids", [])) + len(obj.get("output_ids", []))
        else:
            total_tokens += len(obj.origin_input_ids) + len(obj.output_ids)
    total_tokens /= 2
    print(f"Generation completed in {end_time - start_time:.2f} seconds, total_tokens: {total_tokens}, speed: {total_tokens/(end_time - start_time):.2f}.")

    if result is not None:
        return
    else:
        return


if __name__ == "__main__":
    if torch.cuda.is_available():
        pass
    else:
        print("WARNING: CUDA not available. SGLang dynamic mode will likely fail.")

    mp.set_start_method("spawn", force=True)
    main()
