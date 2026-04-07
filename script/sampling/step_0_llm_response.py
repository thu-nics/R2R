import argparse
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Mapping, MutableMapping

from utils import extract_top_candidates_from_entry


@dataclass
class ResultItem:
    id: str
    input_text: str
    input_ids: list
    output_ids: list
    is_finished: bool
    finish_reason: str | None = None
    top_probs: list | None = None
    top_prob_token_ids: list | None = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate model responses for a dataset with SGLang."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/share/public/public_models/Qwen3-8B",
        help="Path or name of the model to use.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="nics-efc/R2R_query",
        help="Path to the dataset.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to load.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save results.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8192,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Temperature for generation.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Top-k sampling parameter.",
    )
    parser.add_argument(
        "--min_p",
        type=float,
        default=0.0,
        help="Min-p sampling parameter.",
    )
    parser.add_argument(
        "--top_logprobs_num",
        type=int,
        default=20,
        help="Number of top logprobs to save for each generated token.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Data type for model weights.",
    )
    parser.add_argument(
        "--mem_fraction_static",
        type=float,
        default=0.5,
        help="Memory fraction for static allocation in SGLang.",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="Tensor parallelism size for SGLang.",
    )
    parser.add_argument(
        "--is_print",
        action="store_true",
        default=False,
        help="Print all model responses to standard output.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode on the first item only.",
    )
    parser.add_argument(
        "--num_items",
        type=int,
        default=None,
        help="Process only the first N items.",
    )
    return parser.parse_args()


def initialize_sglang_engine(
    model_path, dtype="bfloat16", mem_fraction_static=0.5, tp_size=1
):
    import sglang as sgl
    import r2r.models.sglang_patch.sgl_engine_patcher
    from transformers import AutoTokenizer

    print(f"Initializing SGLang engine from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    engine = sgl.Engine(
        model_path=model_path,
        dtype=dtype,
        mem_fraction_static=mem_fraction_static,
        skip_tokenizer_init=True,
        tp_size=tp_size,
    )
    return engine, tokenizer


def load_items(dataset_path: str, split: str):
    from datasets import load_dataset

    dataset = load_dataset(dataset_path, split=split)
    items = list[Any | Mapping | list | MutableMapping | dict](dataset)
    print(f"Loaded {len(items)} items from {dataset_path} ({split})")
    return items


def limit_items(items, debug: bool, num_items: int | None):
    if debug:
        print("Debug mode: processing only the first item")
        return items[:1]
    if num_items is not None:
        print(f"Processing first {num_items} items")
        return items[:num_items]
    return items


def build_prompt_records(items, tokenizer):
    prompt_records = []
    for item in items:
        item_id = str(item["id"])
        input_text = item["question"]
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": input_text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_records.append(
            {
                "id": item_id,
                "input_text": input_text,
                "prompt": prompt,
                "input_ids": input_ids,
            }
        )
    return prompt_records


def _logprob_to_prob(logprob: float) -> float:
    return math.exp(float(logprob))


def _build_saved_sampling_metadata(
    raw_top_logprobs,
    *,
    max_candidates: int,
):
    steps = (
        list(raw_top_logprobs.values())
        if isinstance(raw_top_logprobs, dict)
        else list(raw_top_logprobs)
    )
    saved_top_probs = []
    saved_top_token_ids = []

    for step_idx, step_entry in enumerate(steps):
        token_ids, scores = extract_top_candidates_from_entry(
            step_entry, max_candidates=max_candidates
        )
        probs = [_logprob_to_prob(score) for score in scores]
        saved_token_ids = [int(token_id) for token_id in token_ids]

        saved_top_token_ids.append(saved_token_ids)
        saved_top_probs.append(
            [
                {"token_id": int(token_id), "prob": float(prob)}
                for token_id, prob in zip(saved_token_ids, probs)
            ]
        )

    return saved_top_probs, saved_top_token_ids


def batch_generate(
    engine,
    prompt_records,
    max_new_tokens=8192,
    temperature=0.0,
    top_p=1.0,
    top_k=20,
    min_p=0.0,
    top_logprobs_num=20,
):
    if not prompt_records:
        return []

    input_ids_list = [record["input_ids"] for record in prompt_records]
    sampling_params = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
    }
    if top_p < 1.0:
        sampling_params["top_p"] = top_p
    if top_k > 0:
        sampling_params["top_k"] = top_k
    if float(min_p) > 0.0:
        sampling_params["min_p"] = min_p

    outputs = engine.generate(
        input_ids=input_ids_list,
        sampling_params=sampling_params,
        return_logprob=top_logprobs_num > 0,
        top_logprobs_num=top_logprobs_num,
    )

    max_candidates = min(int(top_logprobs_num), 20)

    results = []
    for output in outputs:
        meta_info = output["meta_info"]
        raw_top_logprobs = meta_info["output_top_logprobs"]
        output_ids = [int(token_id) for token_id in output["output_ids"]]
        saved_top_probs, saved_top_token_ids = _build_saved_sampling_metadata(
            raw_top_logprobs,
            max_candidates=max_candidates,
        )

        results.append(
            {
                "output_ids": output_ids,
                "finish_reason": meta_info["finish_reason"],
                "top_probs": saved_top_probs,
                "top_prob_token_ids": saved_top_token_ids,
            }
        )

    return results


def process_batch(prompt_records, responses, tokenizer=None, is_print=False):
    results = []
    for record, generation_output in zip(prompt_records, responses):
        finish_reason = generation_output["finish_reason"]
        is_finished = finish_reason == "stop"

        if is_print and tokenizer is not None:
            prompt_text = tokenizer.decode(record["input_ids"], skip_special_tokens=False)
            output_text = tokenizer.decode(generation_output["output_ids"], skip_special_tokens=False)
            print(f"\n===== FORMATTED PROMPT =====\n{prompt_text}\n")
            print(f"===== FULL OUTPUT =====\n{output_text}\n")
            print(f"{'=' * 50}\n")

        results.append(
            ResultItem(
                id=record["id"],
                input_text=record["input_text"],
                input_ids=record["input_ids"],
                output_ids=generation_output["output_ids"],
                is_finished=is_finished,
                finish_reason=finish_reason,
                top_probs=generation_output["top_probs"],
                top_prob_token_ids=generation_output["top_prob_token_ids"],
            )
        )
    return results


def save_results(results, output_dir):
    import pandas as pd
    from datasets import Dataset

    os.makedirs(output_dir, exist_ok=True)

    if not results:
        print("No results to save.")
        return

    rows = [asdict(result) for result in results]
    df = pd.DataFrame(rows)

    csv_path = os.path.join(output_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV results to: {csv_path}")

    dataset_path = os.path.join(output_dir, "dataset")
    Dataset.from_list(rows).save_to_disk(dataset_path)
    print(f"Saved dataset to: {dataset_path}")


def main():
    args = parse_args()
    items = load_items(args.dataset_path, args.split)
    items = limit_items(items, args.debug, args.num_items)

    if not items:
        print("No items to process.")
        return

    print(f"Processing {len(items)} items")
    print(f"Output directory: {args.output_dir}")

    engine, tokenizer = initialize_sglang_engine(
        model_path=args.model_path,
        dtype=args.dtype,
        mem_fraction_static=args.mem_fraction_static,
        tp_size=args.tp_size,
    )

    try:
        results = []
        total_batches = (len(items) + args.batch_size - 1) // args.batch_size

        for batch_index in range(0, len(items), args.batch_size):
            batch = items[batch_index : batch_index + args.batch_size]
            current_batch = batch_index // args.batch_size + 1
            print(f"Processing batch {current_batch}/{total_batches}")

            prompt_records = build_prompt_records(batch, tokenizer)
            responses = batch_generate(
                engine=engine,
                prompt_records=prompt_records,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                top_logprobs_num=args.top_logprobs_num,
            )
            results.extend(
                process_batch(
                    prompt_records=prompt_records,
                    responses=responses,
                    tokenizer=tokenizer,
                    is_print=args.is_print,
                )
            )

        save_results(results, args.output_dir)
    finally:
        print("Shutting down SGLang engine")
        try:
            engine.shutdown()
        except Exception as exc:
            print(f"Error shutting down engine: {exc}")

    print("All processing complete!")


if __name__ == "__main__":
    main()
