"""
This script is used to convert different huggingface datasets with query and answer to a unified dataset for LLM processing.

The names of input dataset are expected to be supported in the support_dataset_config.json.

The output is a unified dataset in the output directory.
"""

import os
import json
import argparse
from datasets import load_dataset, load_from_disk, Dataset
from tqdm import tqdm
import pandas as pd
from r2r.evaluate.eval_utils import prepare_multiple_choice_prompt
from r2r.utils.dataset_conversion import BeSpokeStratosDataset

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert datasets to a unified format for LLM processing"
    )
    
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="Dataset configuration name(s) to use, comma-separated for multiple configs"
    )
    
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Default dataset split to use if not specified in config"
    )
    
    parser.add_argument(
        "--is_local",
        action="store_true",
        help="Use local dataset from disk instead of HuggingFace"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="unified_datasets",
        help="Directory to save the converted dataset"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (only process first few items)"
    )
    
    parser.add_argument(
        "--num_items",
        type=int,
        default=None,
        help="Number of items to process (for testing)"
    )
    
    return parser.parse_args()

def load_dataset_config(dataset_config_name):
    """Load dataset configuration from JSON file."""
    config_file = os.path.join(os.path.dirname(__file__), "support_dataset_config.json")
    try:
        with open(config_file, "r") as f:
            dataset_config = json.load(f)
            if dataset_config_name in dataset_config:
                return dataset_config[dataset_config_name]
            else:
                raise ValueError(f"Dataset config {dataset_config_name} not found in {config_file}")
    except Exception as e:
        print(f"Warning: Could not load dataset config from {config_file}. Error: {e}")
        return {
            "default": {
                "id_field": "id",
                "message_format": [{"role": "user", "content_field": "content"}],
                "add_generation_prompt": True,
            }
        }

def detect_dataset_config(dataset_path, dataset_config, specified_config=None):
    """Determine which dataset configuration to use based on dataset path or specified config."""
    if specified_config and specified_config in dataset_config:
        return specified_config
    
    # Try to auto-detect from dataset path
    dataset_path = str(dataset_path)
    for key in dataset_config:
        if key != "default" and key in dataset_path:
            return key
    
    return "default"

def convert_dataset(args, dataset, config):
    """Convert a dataset to unified format based on configuration."""
    converted_data = []
    
    for idx, item in enumerate(tqdm(dataset, desc="Converting dataset")):
        try:
            # Extract ID or create one if not present
            item_id = item.get(config["id_field"], str(idx))
            
            # Format input message based on configuration
            formatted_input = ""
            for msg_config in config["message_format"]:
                role = msg_config["role"]
                content_field = msg_config["content_field"]
                
                # Get content from the specified field, with empty string as fallback
                content = item.get(content_field, "")
                
                # Handle special query formatting if specified
                if config.get("query_format"):
                    query_type = config["query_format"]["query_type"]
                    format_config = config["query_format"]
                    if query_type == "multiple_choice":
                        content, _ = prepare_multiple_choice_prompt(item, format_config)
                
                if content:  # Only add message if content is not empty
                    formatted_input = content
                    break  # Use the first non-empty content field
            
            # Create unified item
            unified_item = {
                "id": item_id,
                "original_data": item,
                "question": formatted_input,
                "source": config["_dataset_source"],
                "type": config["type"]
            }
            
            converted_data.append(unified_item)
            
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            continue
    
    return converted_data

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse comma-separated dataset configs
    dataset_configs = [config.strip() for config in args.dataset_config.split(",")]
    
    all_converted_data = []
    
    # Process each dataset with its corresponding config from the JSON file
    for config_name in dataset_configs:
        print(f"Processing config: {config_name}")
        
        # Load dataset configuration
        dataset_config = load_dataset_config(config_name)
        print(f"Dataset config: {dataset_config}")
        
        # Get dataset path, subset, and split from the config
        dataset_path = dataset_config.get("dataset_path")
        if not dataset_path:
            print(f"Error: dataset_path not found in config {config_name}")
            continue
            
        dataset_subset = dataset_config.get("dataset_subset")
        dataset_split = dataset_config.get("dataset_split", args.dataset_split)
        
        print(f"Dataset path: {dataset_path}, subset: {dataset_subset}, split: {dataset_split}")
        
        # Load the dataset
        try:
            if args.is_local:
                dataset = load_from_disk(dataset_path)
                if dataset_split and dataset_split in dataset:
                    dataset = dataset[dataset_split]
            else:
                if dataset_subset:
                    dataset = load_dataset(
                        dataset_path, dataset_subset, split=dataset_split
                    )
                else:
                    dataset = load_dataset(dataset_path, split=dataset_split)
        except Exception as e:
            print(f"Error loading dataset {dataset_path}: {e}")
            continue
        
        if dataset_path == "bespokelabs/Bespoke-Stratos-17k":
            dataset = BeSpokeStratosDataset().filter_dataset(dataset, dataset_config["filter"])
        
        # Convert to list for processing
        all_items = list(dataset)
        
        # Handle debug mode and num_items
        if args.debug:
            all_items = all_items[:10]
            print("Debug mode: processing only first 10 items")
        elif args.num_items:
            all_items = all_items[:args.num_items]
            print(f"Processing first {args.num_items} items")
        
        # Add dataset source information to config
        dataset_config["_dataset_source"] = dataset_path
        
        # Convert the dataset
        converted_data = convert_dataset(args, all_items, dataset_config)
        
        if not converted_data:
            print(f"No data was converted for {dataset_path}!")
            continue
        
        all_converted_data.extend(converted_data)
    
    if not all_converted_data:
        print("No data was converted from any dataset!")
        return
    
    # Create a combined Dataset object from all converted data
    unified_dataset = Dataset.from_dict({
        "id": [item["id"] for item in all_converted_data],
        "question": [item["question"] for item in all_converted_data],
        'source': [item["source"] for item in all_converted_data],
        'type': [item["type"] for item in all_converted_data],
    })
    
    # Save the combined dataset with a generic name
    # output_path = os.path.join(args.output_dir, "combined_dataset")
    output_path = args.output_dir
    unified_dataset.save_to_disk(output_path)
    print(f"Saved unified dataset with {len(unified_dataset)} items to {output_path}")

if __name__ == "__main__":
    main()
