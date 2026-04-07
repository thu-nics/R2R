import json
import os
import argparse
from dataclasses import dataclass
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import pandas as pd
from datasets import load_from_disk

from r2r.utils.sampling import sample_token


class TOKEN_TYPE:
    INPUT_INSTRUCTION = 0
    REASONING = 1
    RESPONSE = 2


RESULT_BUFFER_KEYS = (
    "predictions",
    "real_tokens",
    "token_ids",
    "data_ids",
    "token_types",
    "hidden_states",
    "top_logits",
    "top_logits_indices",
)


@dataclass
class PrefillSampleResult:
    predictions: torch.Tensor
    real_tokens: torch.Tensor
    token_ids: torch.Tensor
    data_ids: torch.Tensor
    token_types: torch.Tensor
    hidden_states: torch.Tensor
    top_logits: torch.Tensor
    top_logits_indices: torch.Tensor


def load_model(model_name):
    model_config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=model_config,
        device_map="auto",
        torch_dtype=torch.float16,
    ).eval()
    print(f"Model {model_name} loaded successfully!")
    return model


def get_prefill_ids(sample):
    return sample["input_ids"] + sample["output_ids"]


def _matches_subsequence(sequence, pattern, start_index):
    if not pattern:
        return False
    end_index = start_index + len(pattern)
    if end_index > len(sequence):
        return False
    return sequence[start_index:end_index] == pattern


def categorize_token_types(token_id_list, tokenizer):
    """
    Categorize tokens into INPUT_INSTRUCTION (0), REASONING (1), or RESPONSE (2)
    """
    think_start_ids = tokenizer.encode("<think>", add_special_tokens=False)
    think_end_ids = tokenizer.encode("</think>", add_special_tokens=False)

    token_types = []
    current_type = TOKEN_TYPE.INPUT_INSTRUCTION

    index = 0
    while index < len(token_id_list):
        if _matches_subsequence(token_id_list, think_start_ids, index):
            current_type = TOKEN_TYPE.REASONING
            token_types.extend([current_type] * len(think_start_ids))
            index += len(think_start_ids)
            continue

        if _matches_subsequence(token_id_list, think_end_ids, index):
            current_type = TOKEN_TYPE.RESPONSE
            token_types.extend([current_type] * len(think_end_ids))
            index += len(think_end_ids)
            continue

        token_types.append(current_type)
        index += 1

    return token_types


def sample_token_batched_sharded(
    logits,
    temperature=1.0,
    top_p=1.0,
    top_k=-1,
    min_p=0.0,
    shard_size=10000,
):
    batch_size = logits.shape[0]

    if batch_size <= shard_size:
        return sample_token(
            logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )

    results = []
    for i in range(0, batch_size, shard_size):
        end_idx = min(i + shard_size, batch_size)
        shard_logits = logits[i:end_idx]
        shard_predictions = sample_token(
            shard_logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )
        results.append(shard_predictions)

    return torch.cat(results, dim=0)


def _init_result_buffers():
    return {key: [] for key in RESULT_BUFFER_KEYS}


def _append_result(buffers, sample_result):
    for key in RESULT_BUFFER_KEYS:
        buffers[key].append(getattr(sample_result, key))


def _finalize_result_tensors(buffers):
    return {key: torch.cat(values, dim=0) for key, values in buffers.items()}


def _run_prefill_sample(sample, data_id, model, tokenizer, args):
    prefill_ids = get_prefill_ids(sample)

    if len(prefill_ids) > args.max_input_length:
        print(
            f"Input length {len(prefill_ids)} exceeds max length {args.max_input_length}, skipping"
        )
        return None

    input_ids = torch.tensor([prefill_ids], dtype=torch.long).to(model.device)

    outputs = model(input_ids, output_hidden_states=True)
    logits = outputs.logits
    top_logits, top_logits_indices = torch.topk(logits[0], 100, dim=-1)

    return PrefillSampleResult(
        predictions=sample_token_batched_sharded(
            logits[0],
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            shard_size=3000,
        ).cpu(),
        real_tokens=input_ids[0].cpu(),
        token_ids=torch.arange(0, input_ids.shape[-1], 1).cpu(),
        data_ids=torch.full((input_ids.shape[-1],), data_id, dtype=torch.int64),
        token_types=torch.tensor(
            categorize_token_types(prefill_ids, tokenizer), dtype=torch.int32
        ).cpu(),
        hidden_states=outputs.hidden_states[-1][0].cpu(),
        top_logits=top_logits.float().cpu(),
        top_logits_indices=top_logits_indices.cpu(),
    )


def _save_prefill_outputs(output_path, model_name, tensors, results_file):
    torch.save(tensors["top_logits"], os.path.join(output_path, "SLM_top_logits.pt"))
    torch.save(
        tensors["top_logits_indices"],
        os.path.join(output_path, "SLM_top_logits_indices.pt"),
    )
    torch.save(
        tensors["hidden_states"], os.path.join(output_path, "SLM_hidden_states.pt")
    )

    results_dict = {
        "predictions": tensors["predictions"],
        "token_id": tensors["token_ids"],
        "data_id": tensors["data_ids"],
        "token_type": tensors["token_types"],
        "top_logits": tensors["top_logits"],
        "top_logits_index": tensors["top_logits_indices"],
        "real_token": tensors["real_tokens"],
        "prefill_model": model_name,
    }
    torch.save(results_dict, results_file)


def process_dataset(args):
    """Prefill the LLM rollout (input_ids + output_ids) with the SLM and save outputs."""
    os.makedirs(args.output_path, exist_ok=True)

    model_name = args.prefill_model
    model_path = model_name.rstrip("/").split("/")[-1]
    print(f"Prefill model: {model_name}")
    print(f"Loading local dataset from {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)

    if hasattr(dataset, "keys"):
        if "train" in dataset.keys():
            dataset = dataset["train"]
        elif "test" in dataset.keys():
            dataset = dataset["test"]

    if args.index_range is not None:
        start_idx, end_idx = args.index_range
        dataset = dataset.select(range(start_idx, end_idx))

    print(f"Dataset length: {len(dataset)}")

    results_file = os.path.join(args.output_path, f"results_test_{model_path}.pth")
    csv_path = os.path.join(args.output_path, "prediction_comparison.csv")
    if os.path.exists(results_file) and os.path.exists(csv_path):
        print(f"Results for {model_name} already exist, skipping prefill.")
        return
    if os.path.exists(results_file) and not os.path.exists(csv_path):
        print(f"Found .pth but missing CSV, regenerating prediction_comparison.csv ...")
        results_dict = torch.load(results_file, weights_only=False)
        tensors = {
            "predictions": results_dict["predictions"],
            "real_tokens": results_dict["real_token"],
            "token_ids": results_dict["token_id"],
            "data_ids": results_dict["data_id"],
            "token_types": results_dict["token_type"],
        }
        _save_prediction_comparison_csv(args.output_path, tensors)
        return

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = load_model(model_name)

    result_buffers = _init_result_buffers()

    pbar = tqdm(total=len(dataset), desc=f"Processing {model_path}")
    with torch.no_grad():
        for data_id, sample in enumerate(dataset):
            sample_result = _run_prefill_sample(
                sample=sample,
                data_id=data_id,
                model=model,
                tokenizer=tokenizer,
                args=args,
            )
            if sample_result is not None:
                _append_result(result_buffers, sample_result)
            pbar.update(1)

    pbar.close()

    if not result_buffers["predictions"]:
        print("No valid samples were processed.")
        return

    tensors = _finalize_result_tensors(result_buffers)
    _save_prefill_outputs(
        output_path=args.output_path,
        model_name=model_name,
        tensors=tensors,
        results_file=results_file,
    )

    _save_prediction_comparison_csv(args.output_path, tensors)

    print("All processing completed!")


def _save_prediction_comparison_csv(output_path, tensors):
    df = pd.DataFrame({
        "row_id": range(len(tensors["predictions"])),
        "real_token": tensors["real_tokens"].numpy(),
        "token_id": tensors["token_ids"].numpy(),
        "data_id": tensors["data_ids"].numpy(),
        "token_type": tensors["token_types"].numpy(),
        "SLM_predictions": tensors["predictions"].numpy(),
    })
    csv_path = os.path.join(output_path, "prediction_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"Prediction comparison saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run SLM prefill on LLM rollout and save predictions"
    )
    parser.add_argument(
        "--dataset_path", type=str, default="output/dataset", help="Path to the local dataset"
    )
    parser.add_argument(
        "--prefill_model",
        type=str,
        default="/share/public/public_models/Qwen3-0.6B",
        help="Model path used for prefill.",
    )
    parser.add_argument(
        "--output_path", type=str, default="output/sampling", help="Directory to save output files"
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=32768,
        help="Maximum length of input tokens",
    )
    parser.add_argument(
        "--index_range",
        nargs=2,
        type=int,
        default=None,
        help="Range of dataset samples to process [start_idx, end_idx]",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Temperature for SLM sampling",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Top-k sampling parameter (-1 to disable)",
    )
    parser.add_argument(
        "--min_p",
        type=float,
        default=0.0,
        help="Min-p sampling parameter",
    )
    args = parser.parse_args()

    process_dataset(args)

    with open(os.path.join(args.output_path, "args.json"), "w") as f:
        json.dump(args.__dict__, f)


if __name__ == "__main__":
    main()
