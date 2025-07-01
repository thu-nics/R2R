"""
Step 4: Construct Labeled Dataset for Router Training.

This script combines verification scores (from Step 3) with Small Language Model (SLM)
outputs (top logits, token indices, hidden states) to create a HuggingFace Dataset.
The dataset is used for R2R router training.

Core Functionality:
1. Loads verification scores, SLM outputs, and an overall token index.
2. Filters data by specified `data_id` and `token_id` ranges if provided.
3. Aligns verification scores with SLM data, adjusting `token_id`s and handling missing scores.
4. Removes initial "instruction" tokens based on token_type column (token_type == 0).
5. Filters data to include only `data_id`s present in the verification scores.
6. Constructs and saves a HuggingFace `Dataset` containing token-level features:
   `token_id`, `data_id`, verification score, SLM token, real token, SLM logits,
   SLM logit indices, SLM hidden states, and a divergence indicator. Note that real_token is the input token, others are all corresponding output tokens, thus the i-th real token commonly equals to the (i-1)-th small_token. Real token is only added for reference
7. Saves a 'scalar.csv' with non-tensor data for easy inspection.

Inputs:
- `--csv`: Verification scores CSV from Step 3 (e.g., "verification_results.csv").
- `prediction_comparison.csv`: Index file with `SLM_predictions`, `real_token`, `token_type`, etc.
- `SLM_top_logits.pt`, `SLM_top_logits_indices.pt`, `SLM_hidden_states.pt`: SLM output tensors.
- `--output_sub_folder`: Where to save the dataset.
- `--divergent_column_name`: Column in Step 3 CSV with verification scores.
- Optional filtering args: `--min_data_id`, `--max_data_id`, `--final_max_token_id`.

Outputs:
- HuggingFace `Dataset`: Contains processed, aligned data.
- `scalar.csv`: CSV with scalar features from the dataset.
"""

import os
import pandas as pd
import torch
from datasets import Dataset, Features, Value, Sequence, Array2D, concatenate_datasets
from tqdm import tqdm
import numpy as np
import argparse

from r2r.utils.config import TOKEN_TYPE

parser = argparse.ArgumentParser(description='Prepare dataset for model comparison analysis')

# Data directory and file paths
parser.add_argument('--data_dir', type=str, required=True,
                    help='Directory containing the input data files')
parser.add_argument('--csv', type=str, required=True,
                    help='Name of the input CSV file')
parser.add_argument('--output_sub_folder', type=str, required=True,
                    help='Name of the output subfolder')
parser.add_argument('--divergent_column_name', type=str, required=True,
                    help='Name of the column containing divergent scores')

# Data filtering options
parser.add_argument('--min_data_id', type=int, default=None,
                    help='Minimum data_id to include (inclusive)')
parser.add_argument('--max_data_id', type=int, default=None,
                    help='Maximum data_id to include (inclusive)')
parser.add_argument('--final_max_token_id', type=int, default=None,
                    help='Maximum token_id to include for the final data_id')
parser.add_argument('--batch_size', type=int, default=100000,
                    help='Batch size for dataset creation')

# Processing options
parser.add_argument('--comparison_model', type=str, default='real',
                    choices=['real'],
                    help='Model type to compare against: real')

args = parser.parse_args()

# Update global variables with command line arguments
data_dir = args.data_dir
output_sub_folder = args.output_sub_folder
csv = args.csv
divergent_column_name = args.divergent_column_name
min_data_id = args.min_data_id
max_data_id = args.max_data_id
final_max_token_id = args.final_max_token_id


# Determine the column name to use in the dataset based on the divergent_column_name
if 'divergent' in divergent_column_name:
    dataset_column_name = 'divergent'
else:
    raise NotImplementedError(f"Unknown divergent column name: {divergent_column_name}")

# File paths
df_divergent_score = os.path.join(data_dir, csv)

df_index = os.path.join(data_dir, "prediction_comparison.csv")

small_logits_path = os.path.join(data_dir, "SLM_top_logits.pt")
small_logits_index = os.path.join(data_dir, "SLM_top_logits_indices.pt")
small_last_hidden_states_path = os.path.join(data_dir, "SLM_hidden_states.pt")

def parse_data(output,divergent_column_name):
    output = str(output)
    if 'divergent' in divergent_column_name:
        if '1' in output:
            return 1
        elif '0' in output:
            return 0
        else:
            return -1
    else:
        raise ValueError(f"Unknown divergent column name: {divergent_column_name}")

def load_data():
    """
    Load all data sources and return them as a tuple.
    Returns:
        tuple: (divergent_df, data_index_df, small_logits, small_indices, ref_logits, ref_indices)
    """
    print("Loading CSV files...")
    # Load divergent scores - this contains only the compared tokens
    divergent_df = pd.read_csv(df_divergent_score)
    
    # Get unique data_ids from divergent_df
    unique_data_ids = divergent_df['data_id'].unique()
    print(f"Found {len(unique_data_ids)} unique data_ids in divergent scores")
    
    # Load index data - this contains all tokens and matches with logits
    data_index_df = pd.read_csv(df_index)
    
    print("Loading PyTorch tensors...")
    small_logits = torch.load(small_logits_path, map_location=torch.device('cpu'))
    small_indices = torch.load(small_logits_index, map_location=torch.device('cpu'))
    small_last_hidden_states = torch.load(small_last_hidden_states_path, map_location=torch.device('cpu'))

    # Filter by max_data_id if specified
    if max_data_id is not None:
        print(f"Filtering data to max_data_id: {max_data_id}")
        # Get indices where data_id <= max_data_id
        if min_data_id is not None and max_data_id is not None:
            valid_indices = (data_index_df['data_id'] <= max_data_id) & (data_index_df['data_id'] >= min_data_id)
        elif min_data_id is not None and max_data_id is None:
            valid_indices = (data_index_df['data_id'] >= min_data_id)
        elif min_data_id is None and max_data_id is not None:
            valid_indices = (data_index_df['data_id'] <= max_data_id)
        
        if final_max_token_id is not None:
            if max_data_id == min_data_id:
                valid_indices = (data_index_df['data_id'] == max_data_id) & (data_index_df['token_id'] <= final_max_token_id)
            else:
                valid_indices = ((data_index_df['data_id'] <= max_data_id - 1) & (data_index_df['data_id'] >= min_data_id)) | ((data_index_df['data_id'] == max_data_id) & (data_index_df['token_id'] <= final_max_token_id))
        
        valid_indices_tensor = torch.tensor(valid_indices.values, dtype=torch.bool)
        
        # Filter all data sources
        if len(data_index_df) == small_logits.shape[0]:
            small_logits = small_logits[valid_indices_tensor]
            small_indices = small_indices[valid_indices_tensor]
            small_last_hidden_states = small_last_hidden_states[valid_indices_tensor]
        elif len(data_index_df) < small_logits.shape[0]:
            print(f"Data length mismatch: data_index_df={len(data_index_df)}, small_logits={small_logits.shape[0]}")
            small_logits = small_logits[valid_indices_tensor]
            small_indices = small_indices[valid_indices_tensor]
            small_last_hidden_states = small_last_hidden_states[valid_indices_tensor]
        else:
            raise ValueError(f"Data length mismatch: data_index_df={len(data_index_df)}, small_logits={small_logits.shape[0]}")
        
        data_index_df = data_index_df[valid_indices]
        
        print(f"Filtered to {len(data_index_df)} rows")
    
    # Verify tensor shapes match index df
    assert len(data_index_df) == len(small_logits) == len(small_last_hidden_states), \
        f"Data length mismatch: data_index_df={len(data_index_df)}, small_logits={len(small_logits)}, small_last_hidden_states={len(small_last_hidden_states)}"
    
    small_last_hidden_states = small_last_hidden_states.to(dtype=torch.float32)

    print(f"Total rows: {len(data_index_df)}")
    print(f"Rows with divergent scores: {len(divergent_df)}")
    print(f"Unique data_ids: {data_index_df['data_id'].nunique()}")
    
    return divergent_df, data_index_df, small_logits, small_indices, small_last_hidden_states, unique_data_ids

def align_data(divergent_df, data_index_df):
    """
    Align divergent scores with the full dataset, filling missing values with 0.
    Adjusts token_ids in divergent_df by subtracting 1 to match df_index.
    """
    print("Aligning data...")
    # Keep only necessary columns from divergent df and adjust token_id
    divergent_df_updated = divergent_df[['data_id', 'token_id', divergent_column_name]].copy()
    divergent_df_updated['token_id'] = divergent_df_updated['token_id'] - 1  # Adjust token_id to match df_index
    
    # Convert divergent scores to numeric values
    divergent_df_updated[divergent_column_name] = pd.to_numeric(divergent_df_updated[divergent_column_name], errors='coerce')
    
    divergent_df_updated['mismatch'] = 1
    
    # Merge with index df using both data_id and token_id
    merged_df = pd.merge(
        data_index_df,
        divergent_df_updated,
        on=['data_id', 'token_id'],
        how='left'
    )
    
    # Fill missing values based on the column type
    if 'divergent' in divergent_column_name:
        # For divergent column, fill missing values with 0 (not divergent)
        merged_df[divergent_column_name] = merged_df[divergent_column_name].fillna(0)
        merged_df['mismatch'] = merged_df['mismatch'].fillna(0)
        print(f"Data aligned. Tokens marked as divergent (1): {(merged_df[divergent_column_name] == 1).sum()}")
    else:
        raise NotImplementedError(f"Unknown divergent column name: {divergent_column_name}")
    
    # Convert to numeric again after merge to ensure proper type
    merged_df[divergent_column_name] = pd.to_numeric(merged_df[divergent_column_name], errors='coerce')
    
    # Use token_type column to identify instruction tokens instead of think token
    # token_type == 0 corresponds to INPUT_INSTRUCTION tokens
    if 'token_type' not in merged_df.columns:
        raise ValueError("token_type column not found in data. Make sure Step 1 was run with the latest version that includes token_type.")
    
    # Create mask directly from token_type: 0 for instruction tokens, 1 for non-instruction tokens
    valid_mask = (merged_df['token_type'] != TOKEN_TYPE.INPUT_INSTRUCTION).astype(int)
    
    # Create mask column: 0 for instruction tokens, 1 for non-instruction tokens (reasoning + response)
    merged_df['mask'] = valid_mask
    
    print(f"Instruction tokens (mask=0): {(merged_df['mask'] == 0).sum()}")
    print(f"Non-instruction tokens (mask=1): {(merged_df['mask'] == 1).sum()}")

    # if len(think_token_location[0]) != len(data_id_start[0]):
    #     raise ValueError("think_token_location and data_id_start have different lengths")

    print(f"original data length: {len(merged_df)}")

    return merged_df, valid_mask

def create_dataset(batch_size=100000):
    """
    Create a HuggingFace dataset from the loaded data in batches.
    
    Args:
        batch_size: Number of samples to process in each batch
        
    Returns:
        Dataset: HuggingFace dataset with aligned data
    """
    # Load all data
    divergent_df, data_index_df, small_logits, small_indices, small_last_hidden_states, unique_data_ids = load_data()
    
    # Filter data_index_df to only include rows with data_ids in divergent_df
    print(f"Filtering data to only include rows with data_ids in divergent scores...")
    data_filter = data_index_df['data_id'].isin(unique_data_ids)
    filtered_data_index_df = data_index_df[data_filter]
    
    # Get indices of filtered rows to apply to tensors
    filtered_indices = data_filter.to_numpy().nonzero()[0]
    
    # Filter tensors
    filtered_small_logits = small_logits[filtered_indices]
    filtered_small_indices = small_indices[filtered_indices]
    filtered_small_last_hidden_states = small_last_hidden_states[filtered_indices]

    print(f"Filtered data from {len(data_index_df)} to {len(filtered_data_index_df)} rows")
    
    # Align data using filtered_data_index_df as base
    aligned_df, mask = align_data(divergent_df, filtered_data_index_df)

    # Define features for the dataset

    features = Features({
        'token_id': Value('int64'),
        'data_id': Value('int64'),
        dataset_column_name: Value('int64'),
        'small_token': Value('int64'),
        'real_token': Value('int64'),
        'small_logits': Sequence(feature=Value('float32')),
        'small_indices': Sequence(feature=Value('int64')),
        'small_last_hidden_states': Sequence(feature=Value('float32')),
        'mismatch': Value('int64'),
        'mask': Value('int64'),
    })
    
    # Calculate number of batches
    total_samples = len(aligned_df)
    num_batches = (total_samples + batch_size - 1) // batch_size  # Ceiling division
    
    print(f"Creating dataset in {num_batches} batches of size {batch_size}...")
    
    # Create dataset in batches
    datasets = []
    dataset_dict = {}  # For analysis purposes
    
    for batch_idx in tqdm(range(num_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_samples)
        
        # Create batch dictionary
        batch_dict = {
            'token_id': aligned_df['token_id'].iloc[start_idx:end_idx].tolist(),
            'data_id': aligned_df['data_id'].iloc[start_idx:end_idx].tolist(),
            dataset_column_name: aligned_df[divergent_column_name].iloc[start_idx:end_idx].tolist(),
            'small_token': aligned_df['SLM_predictions'].iloc[start_idx:end_idx].tolist(),
            'real_token': aligned_df['real_token'].iloc[start_idx:end_idx].tolist(),
            'small_logits': filtered_small_logits[start_idx:end_idx].numpy(),
            'small_indices': filtered_small_indices[start_idx:end_idx].numpy(),
            'small_last_hidden_states': filtered_small_last_hidden_states[start_idx:end_idx].numpy(),
            'mismatch': aligned_df['mismatch'].iloc[start_idx:end_idx].tolist(),
            'mask': mask[start_idx:end_idx].tolist(),
        }
        
        # Create dataset for this batch
        batch_dataset = Dataset.from_dict(batch_dict)
        batch_dataset = batch_dataset.cast(features)
        datasets.append(batch_dataset)
        
        # Store the first batch for analysis
        if batch_idx == 0:
            dataset_dict = batch_dict.copy()
    
    # Concatenate all batch datasets
    print("Concatenating batched datasets...")
    dataset = concatenate_datasets(datasets)
    
    # For analysis purposes, update dataset_dict with full data
    dataset_dict = {
        'token_id': aligned_df['token_id'].tolist(),
        'data_id': aligned_df['data_id'].tolist(),
        dataset_column_name: aligned_df[divergent_column_name].tolist(),
        'small_token': aligned_df['SLM_predictions'].tolist(),
        'real_token': aligned_df['real_token'].tolist(),
        'small_logits': filtered_small_logits.numpy(),
        'small_indices': filtered_small_indices.numpy(),
        'small_last_hidden_states': filtered_small_last_hidden_states.numpy(),
        'mismatch': aligned_df['mismatch'].tolist(),
        'mask': mask.tolist(),
    }
    
    print(f"Dataset created with {len(dataset)} samples")
    return dataset, dataset_dict

def save_dataset(dataset, output_path):
    """
    Save the dataset to disk.
    Args:
        dataset: HuggingFace dataset to save
        output_path: Path to save the dataset
    """
    dataset.save_to_disk(output_path)
    print(f"Dataset saved to {output_path}")

def analyze_dataset(dataset_dict):
    """
    Analyze and print basic information about the aligned dataset.
    Args:
        dataset_dict: Dictionary containing the dataset
    """
    print("\n===== Dataset Statistics =====")
    
    # Basic counts
    num_rows = len(dataset_dict['token_id'])
    print(f"Total number of rows: {num_rows}")
    
    # Data ID range
    data_ids = dataset_dict['data_id']
    min_data_id = min(data_ids)
    max_data_id = max(data_ids)
    unique_data_ids = len(set(data_ids))
    print(f"Data ID range: {min_data_id} to {max_data_id}")
    print(f"Number of unique data IDs: {unique_data_ids}")
    
    # Token ID statistics
    token_ids = dataset_dict['token_id']
    min_token_id = min(token_ids)
    max_token_id = max(token_ids)
    print(f"Token ID range: {min_token_id} to {max_token_id}")
    
    # divergent/divergent score distribution
    scores = np.array(dataset_dict[dataset_column_name])
    score_bins = [0, 1, 2]
    hist, _ = np.histogram(scores, bins=score_bins)
    
    print(f"\n{dataset_column_name.capitalize()} score distribution:")
    for i in range(len(score_bins)-1):
        score = score_bins[i]
        count = hist[i]
        percentage = (count / num_rows) * 100
        print(f"  {score}: {count} rows ({percentage:.2f}%)")
    
    if dataset_column_name == 'divergent':
        fill_value = 0
        print(f"\nRows with filled {dataset_column_name} score ({fill_value}): {(scores == fill_value).sum()} ({(scores == fill_value).sum() / num_rows * 100:.2f}%)")
        print(f"Rows with actual {dataset_column_name} scores: {(scores != fill_value).sum()} ({(scores != fill_value).sum() / num_rows * 100:.2f}%)")
    
    # Print shapes of non-scalar columns (tensors/arrays)
    print("\nShapes of non-scalar columns:")
    for key, value in dataset_dict.items():
        if isinstance(value, (np.ndarray, list)) and len(value) > 0:
            if isinstance(value, list):
                # For lists, check if the first element is a scalar or not
                if isinstance(value[0], (int, float, str, bool)):
                    continue  # Skip scalar lists
                else:
                    # Try to get shape of the first element
                    try:
                        first_elem_shape = np.array(value[0]).shape
                        print(f"  {key}: List of {len(value)} items, each with shape {first_elem_shape}")
                    except:
                        print(f"  {key}: List of {len(value)} items (non-scalar)")
            else:  # numpy array
                print(f"  {key}: {value.shape}")
    
    print("=============================\n")

if __name__ == "__main__":
    # Create and save the dataset
    output_path = os.path.join(data_dir, output_sub_folder)
    print(f"Creating dataset...")
    dataset, dataset_dict = create_dataset(batch_size=args.batch_size)  # Use batch size from command line args
    
    # Analyze the dataset
    analyze_dataset(dataset_dict)
    
    print(f"Saving dataset to {output_path}...")
    save_dataset(dataset, output_path)
    
    # Save scalar values to CSV
    print(f"Saving scalar values to {output_path}/scalar.csv...")
    
    scalar_dict = {
        'token_id': dataset_dict['token_id'],
        'data_id': dataset_dict['data_id'],
        dataset_column_name: dataset_dict[dataset_column_name],
        'small_token': dataset_dict['small_token'],
        'real_token': dataset_dict['real_token'],
        'mismatch': dataset_dict['mismatch'],
        'mask': dataset_dict['mask'],
    }
    df_scalar = pd.DataFrame(scalar_dict)
    df_scalar = df_scalar.sort_values(by=['data_id', 'token_id'])
    df_scalar.to_csv(os.path.join(output_path, "scalar.csv"), index=False)
    
    print("Done!")
