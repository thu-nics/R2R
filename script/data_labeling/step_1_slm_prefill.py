"""
Step1 use SLM to prefill the LLM responses, finding all non-identical SLM next-token predictions.

Inputs:
- A huggingface dataset with the model responses.
    - The dataset from Step0. It contains columns: "id", "input_text", "model_reasoning", "model_response", and "is_finished". Each row corresponds to a query.

Outputs:
- prediction_comparison.csv: A csv file comparing LLM and SLM next-token predictions 
    - contains columns: data_id, token_id, real_token (predited tokens from LLM),SLM_predictions,SLM_prediction_samples
    - each row corresponds to a token in the original LLM response
- SLM_hidden_states/top_logits/top_logits_indices.pt: The last-layer hidden states, top logits, and top logits indices of the SLM for each token prediction. All tensors have the same first dimension of #total_tokens.
"""

import json
import os
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from datasets import load_from_disk
import pandas as pd
import glob
import numpy as np

from r2r.utils.config import TOKEN_TYPE, MODEL_DICT
from r2r.utils.sampling import sample_token

def load_model(model_name):
    """Load a model with basic error handling"""
    try:
        model_config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=model_config,
            device_map="auto",
            torch_dtype=torch.float16,
        ).eval()
        print(f"Model {model_name} loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def apply_qwen_r1_chat_template(messages, add_generation_prompt=False):
    """
    Apply the Qwen R1 chat template to the messages. We rewrite the function to use the same template as the original one, adding the thinking process in the context. 
    The thinking process is originally excluded for multi-turn conversations.
    """
    prompt = "<｜begin▁of▁sentence｜>"
    ns = {
        "is_first": False,
        "is_tool": False,
        "is_output_first": True,
        "system_prompt": "",
    }

    # extract system prompt
    for message in messages:
        if message["role"] == "system":
            ns["system_prompt"] = message["content"]

    prompt += ns["system_prompt"]

    for message in messages:
        if message["role"] == "user":
            ns["is_tool"] = False
            prompt += "<｜User｜>" + message["content"]

        elif message["role"] == "assistant" and message["content"] is not None:
            content = message["content"]
            prompt += "<｜Assistant｜>" + content + "<｜end▁of▁sentence｜>"

    if add_generation_prompt:
        prompt += "<｜Assistant｜><think>\n"

    return prompt

def get_formatted_prompt(sample, dataset_path, tokenizer, model_name):
    """Format prompt based on dataset type"""
    input_text = sample["input_text"]
    model_reasoning = sample["model_reasoning"]
    model_response = sample["model_response"]

    if model_reasoning == None or model_response == None:
        print(f"model_reasoning or model_response is None, skip")
        return None
    input_text = sample["input_text"]

    messages = [
        {"role": "user", "content": input_text},
        {
            "role": "assistant",
            "content": None,
        },
    ]
    
    if "r1" in model_name.lower():
        messages[1]["content"] = f"<think>\n{model_reasoning}\n</think>\n\n" + model_response
        prompt = apply_qwen_r1_chat_template(messages, add_generation_prompt=False)
    else:
        messages[1]["content"] = f"{model_reasoning}\n</think>\n\n" + model_response
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=True)
    return prompt


def categorize_token_types(input_ids, tokenizer):
    """
    Categorize tokens into INPUT_INSTRUCTION (0), REASONING (1), or RESPONSE (2)
    
    Args:
        input_ids: torch tensor of token IDs
        tokenizer: tokenizer used for encoding
    
    Returns:
        List of token type categories
    """
    THINK_START_TOKEN = MODEL_DICT["special_tokens"]["think_start"]
    THINK_END_TOKEN = MODEL_DICT["special_tokens"]["think_end"]
    
    token_types = []
    current_type = TOKEN_TYPE.INPUT_INSTRUCTION
    
    for i, token_id in enumerate(input_ids[0]):
        token_id = token_id.item()
        
        if token_id == THINK_START_TOKEN:
            current_type = TOKEN_TYPE.REASONING
        elif token_id == THINK_END_TOKEN:
            current_type = TOKEN_TYPE.RESPONSE
            
        token_types.append(current_type)
    
    return token_types


def sample_token_batched_sharded(logits, temperature=1.0, top_p=1.0, top_k=-1, shard_size=10000):
    """
    Sample tokens from batched logits with sharding for large batch sizes.
    
    Args:
        logits: Tensor of shape [batch_size, vocab_size]
        temperature: Temperature for sampling
        top_p: Top-p probability threshold for nucleus sampling
        top_k: Top-k threshold for sampling
        shard_size: Maximum size of each shard (default: 3000)
        
    Returns:
        Tensor of sampled token IDs for the entire batch
    """
    batch_size = logits.shape[0]
    
    # If batch size is smaller than shard size, process directly
    if batch_size <= shard_size:
        return sample_token(logits, temperature=temperature, top_p=top_p, top_k=top_k)
    
    # Split into shards and process each one
    results = []
    for i in range(0, batch_size, shard_size):
        end_idx = min(i + shard_size, batch_size)
        shard_logits = logits[i:end_idx]
        
        # Sample from this shard
        shard_predictions = sample_token(shard_logits, temperature=temperature, top_p=top_p, top_k=top_k)
        results.append(shard_predictions)
    
    # Concatenate all results
    return torch.cat(results, dim=0)


def process_dataset(args):
    """Process the dataset with all models and directly create the final prediction_comparison.csv"""
    # Create output directory
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Get model sizes for organizing output
    all_model_sizes = [
        float(model.split("-")[-1].strip("/").replace("B", ""))
        for model in args.test_model_list
    ]
    all_model_sizes.sort()
    print(f"Model sizes: {all_model_sizes}")

    # Load dataset
    print(f"Loading local dataset from {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)

    # Handle dataset splits
    if hasattr(dataset, "keys"):
        if "train" in dataset.keys():
            dataset = dataset["train"]
        elif "test" in dataset.keys():
            dataset = dataset["test"]

    # Limit dataset size if specified
    if args.index_range is not None:
        start_idx, end_idx = args.index_range
        dataset = dataset.select(range(start_idx, end_idx))

    print(f"Dataset length: {len(dataset)}")

    # Dictionary to store all predictions per model
    all_predictions = {}
    # Initialize lists to store common data
    all_real_tokens = []
    all_token_ids = []
    all_data_ids = []
    all_token_types = []

    # Process each model
    for model_name in args.test_model_list:
        model_size = float(model_name.split("-")[-1].replace("B", ""))
        model_path = model_name.split("/")[-1]

        # Skip if results already exist
        if os.path.exists(
            os.path.join(args.output_path, f"results_test_{model_path}.pth")
        ):
            print(f"Results for {model_name} already exist, loading from file.")
            results_dict = torch.load(
                os.path.join(args.output_path, f"results_test_{model_path}.pth"),
                weights_only=False,
            )
            all_predictions[model_size] = results_dict["predictions"]
            # Also load common data if we haven't processed any models yet
            if not all_real_tokens:
                all_real_tokens.append(results_dict["real_token"])
                all_token_ids.append(results_dict["token_id"]) 
                all_data_ids.append(results_dict["data_id"])
                all_token_types.append(results_dict["token_type"])
            continue

        # Load tokenizer for this model
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model
        model = load_model(model_name)
        if model is None:
            continue

        # Store results for this model
        predictions_list = []
        real_tokens_list = []
        token_ids_list = []
        data_ids_list = []
        token_types_list = []
        all_hidden_states = []
        all_top_logits = []
        all_top_logits_indices = []

        # Process each sample
        pbar = tqdm(total=len(dataset), desc=f"Processing {model_path}")
        with torch.no_grad():
            for data_id, sample in enumerate(dataset):
                # Get formatted prompt
                prompt = get_formatted_prompt(sample, args.dataset_path, tokenizer, model_name)
                if prompt is None:
                    pbar.update(1)
                    continue

                # Tokenize
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
                    model.device
                )

                # Skip if too long
                if len(input_ids[0]) > args.max_input_length:
                    print(
                        f"Input length {len(input_ids[0])} exceeds max length {args.max_input_length}, skipping"
                    )
                    pbar.update(1)
                    continue

                # Run inference with output_hidden_states=True
                outputs = model(input_ids, output_hidden_states=True)
                logits = outputs.logits
                
                # Use batched sharded sampling instead of single sequence sampling
                pred = sample_token_batched_sharded(logits[0], temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, shard_size=3000)

                pred = pred.cpu()

                # Extract token IDs and data IDs
                token_id = torch.arange(0, input_ids.shape[-1], 1).cpu()
                data_id_tensor = torch.full_like(token_id, data_id).cpu()

                # Extract real tokens
                real_token = input_ids[0].cpu()

                # Categorize token types
                token_types = categorize_token_types(input_ids, tokenizer)
                token_types_tensor = torch.tensor(token_types, dtype=torch.int32).cpu()

                # Extract top logits (top 100 to match small_ref)
                top_logits, top_logits_indices = torch.topk(logits[0], 100, dim=-1)
                # Convert to float32 to match small_ref format
                top_logits = top_logits.float().cpu()
                top_logits_indices = top_logits_indices.cpu()

                # Extract hidden states
                hidden_states = outputs.hidden_states[-1][0].cpu()

                # Append to lists
                predictions_list.append(pred)
                real_tokens_list.append(real_token)
                token_ids_list.append(token_id)
                data_ids_list.append(data_id_tensor)
                token_types_list.append(token_types_tensor)
                all_hidden_states.append(hidden_states)
                all_top_logits.append(top_logits)
                all_top_logits_indices.append(top_logits_indices)

                # pbar.write(f"Model: {model_name} | Input length: {len(input_ids[0])}")
                pbar.update(1)

        pbar.close()

        # Concatenate all results
        predictions = torch.cat(predictions_list, dim=0)
        real_tokens = torch.cat(real_tokens_list, dim=0)
        token_ids = torch.cat(token_ids_list, dim=0)
        data_ids = torch.cat(data_ids_list, dim=0)
        token_types = torch.cat(token_types_list, dim=0)

        # Store predictions in the dictionary
        all_predictions[model_size] = predictions
        all_real_tokens.append(real_tokens)
        all_token_ids.append(token_ids)
        all_data_ids.append(data_ids)
        all_token_types.append(token_types)

        # Save top logits and hidden states
        all_top_logits_tensor = torch.cat(all_top_logits, dim=0)
        all_top_logits_indices_tensor = torch.cat(all_top_logits_indices, dim=0)
        all_hidden_states_tensor = torch.cat(all_hidden_states, dim=0)

        # Save only top logits, indices and hidden states with proper naming
        torch.save(
            all_top_logits_tensor,
            os.path.join(args.output_path, f"SLM_top_logits.pt"),
        )
        torch.save(
            all_top_logits_indices_tensor,
            os.path.join(args.output_path, f"SLM_top_logits_indices.pt"),
        )
        torch.save(
            all_hidden_states_tensor,
            os.path.join(args.output_path, f"SLM_hidden_states.pt"),
        )

        # Also save all in one file to match small_ref format
        results_dict = {
            "predictions": predictions,
            "token_id": token_ids,
            "data_id": data_ids,
            "token_type": token_types,
            "top_logits": all_top_logits_tensor,
            "top_logits_index": all_top_logits_indices_tensor,
            "real_token": real_tokens,
        }
        torch.save(
            results_dict,
            os.path.join(args.output_path, f"results_test_{model_path}.pth"),
        )

    # If we have processed at least one model, combine the common data
    if all_real_tokens:
        real_tokens = torch.cat(all_real_tokens, dim=0)
        token_ids = torch.cat(all_token_ids, dim=0)
        data_ids = torch.cat(all_data_ids, dim=0)
        token_types = torch.cat(all_token_types, dim=0)
    else:
        # Get data from existing results files
        for model_name in args.test_model_list:
            model_path = model_name.split("/")[-1]
            results_file = os.path.join(
                args.output_path, f"results_test_{model_path}.pth"
            )
            if os.path.exists(results_file):
                print(f"Loading data from existing results file: {results_file}")
                results_dict = torch.load(results_file, weights_only=False)
                real_tokens = results_dict["real_token"]
                token_ids = results_dict["token_id"]
                data_ids = results_dict["data_id"]
                token_types = results_dict["token_type"]
                break
        else:
            print("No results files found and no models were processed.")
            return

    # Create data analysis CSV directly
    create_data_analysis(
        output_path=args.output_path,
        model_sizes=all_model_sizes,
        real_tokens=real_tokens,
        token_ids=token_ids,
        data_ids=data_ids,
        token_types=token_types,
        all_predictions=all_predictions,
        top_k=args.top_k,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("All processing completed!")


def create_data_analysis(
    output_path,
    model_sizes,
    real_tokens,
    token_ids,
    data_ids,
    token_types,
    all_predictions,
    top_k=-1,
    temperature=0.6,
    top_p=1.0,
):
    """Create prediction_comparison.csv directly from collected data"""
    # Create DataFrame with common data
    df = pd.DataFrame(
        {
            "row_id": range(len(real_tokens)),
            "real_token": real_tokens.numpy(),
            "token_id": token_ids.numpy(),
            "data_id": data_ids.numpy(),
            "token_type": token_types.numpy(),
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Add predictions from each model
    for model_size in tqdm(model_sizes, desc="Processing model sizes"):
        if model_size in all_predictions:
            df["SLM_predictions"] = all_predictions[model_size].cpu().numpy()

            # Load top logits and indices for this model
            top_logits = torch.load(
                os.path.join(output_path, f"SLM_top_logits.pt"),
                weights_only=False,
                map_location=device,
            )
            top_indices = torch.load(
                os.path.join(output_path, f"SLM_top_logits_indices.pt"),
                weights_only=False,
                map_location=device,
            )

            # Apply temperature and convert to probabilities
            probs = torch.nn.functional.softmax(top_logits / temperature, dim=-1)

            # Vectorized top-p calculation
            # Sort probabilities and get corresponding indices within the top_k dimension
            sorted_probs, indices_in_sorted = torch.sort(probs, dim=-1, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

            # Create mask for top-p
            mask = cumsum_probs <= top_p
            mask[:, 0] = True  # Ensure the top token is always included

            # Create list to store variable-length prediction samples
            all_samples = []

            # Iterate through each row to apply the mask and get final token indices
            for i in tqdm(
                range(probs.shape[0]), desc=f"Processing {model_size} predictions"
            ):
                row_mask = mask[i]
                row_indices_in_sorted = indices_in_sorted[i]
                row_top_indices = top_indices[
                    i
                ]  # Original token indices from the loaded data

                # Get the indices within the sorted list that satisfy the top-p condition
                filtered_indices_in_sorted = row_indices_in_sorted[row_mask]

                # Limit by top_k
                if top_k != -1:
                    k = min(top_k, len(filtered_indices_in_sorted))
                    final_indices_in_sorted = filtered_indices_in_sorted[:k]
                else:
                    final_indices_in_sorted = filtered_indices_in_sorted

                # Map these indices back to the original token IDs using the loaded top_indices
                final_token_ids = row_top_indices[final_indices_in_sorted]

                # Add to list
                all_samples.append(final_token_ids.cpu().tolist())

            # Add variable-length predictions to dataframe
            df["SLM_prediction_samples"] = all_samples

    # Save to CSV
    output_file = os.path.join(output_path, "prediction_comparison.csv")
    df.to_csv(output_file, index=False)
    print(f"Data analysis saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run model inference on datasets and save predictions directly"
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the local dataset"
    )
    parser.add_argument(
        "--test_model_list",
        nargs="+",
        type=str,
        help="List of test models to run",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Directory to save output files"
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
        "--top_k",
        type=int,
        default=-1,
        help="Number of top predictions to include in the output. If -1, no top-k filtering is applied.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature to apply to logits when calculating probabilities",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p probability threshold for nucleus sampling (0 < top_p ≤ 1)",
    )
    args = parser.parse_args()

    if args.test_model_list is None:
        args.test_model_list=[f"{MODEL_DICT['quick']['model_path']}"]
    
    process_dataset(args)

    # save args as json
    with open(os.path.join(args.output_path, "args.json"), "w") as f:
        json.dump(args.__dict__, f)


if __name__ == "__main__":
    main()
