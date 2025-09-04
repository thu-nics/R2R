import os
from pdb import runcall
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk, load_dataset, concatenate_datasets, Value, Sequence
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    recall_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    f1_score,
    accuracy_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from collections import Counter
import json
import argparse
import wandb
from typing import Dict, Any, Optional, Union, List
from tqdm import tqdm
from datetime import datetime
from datasets.features import ClassLabel
from r2r.models.router import create_classifier
from r2r.train.loss import FocalLoss, DPOLoss
from r2r.train.logger import filter_data_id_critical, filter_is_mismatch, filter_has_divergent, print_training_progress, TrainingHistory, create_mask, create_mismatch
from r2r.models.router import save_model, load_model
from r2r.train.optimizer import optimization_pipeline, standard_eval_pipeline

def load_dataset_from_config(dataset_path: str):
    """
    Load dataset from a config path string with format:
    - local:path_to_dataset
    - hf_datasets:key1=value1,key2=value2,...
    
    Args:
        dataset_path: String in format "prefix:kwargs"
        
    Returns:
        Loaded dataset
        
    Examples:
        load_dataset_from_config("local:/path/to/dataset")
        load_dataset_from_config("hf_datasets:path=owner/dataset,split=train")
    """
    if ":" not in dataset_path:
        raise ValueError(f"Invalid dataset path format: {dataset_path}. Expected 'prefix:kwargs'")
    
    prefix, kwargs_str = dataset_path.split(":", 1)
    
    if prefix == "local":
        # For local datasets, the kwargs_str is just the path
        return load_from_disk(kwargs_str)
    
    elif prefix == "hf_datasets":
        # Parse kwargs from comma-separated key=value pairs
        kwargs = {}
        if kwargs_str:
            for pair in kwargs_str.split(","):
                if "=" not in pair:
                    raise ValueError(f"Invalid kwargs format in: {pair}. Expected 'key=value'")
                key, value = pair.split("=", 1)
                kwargs[key.strip()] = value.strip()
        
        return load_dataset(**kwargs)
    
    else:
        raise ValueError(f"Unknown dataset prefix: {prefix}. Supported prefixes: 'local', 'hf_datasets'")

def validate_model(
    model: nn.Module, 
    data_loader: DataLoader, 
    criterion: Any, 
    device: torch.device,
    input_type: List[str] = ["logits"],
    output_type: str = "binary",
    threshold: float = 0.5,
    max_batches: Optional[int] = None
) -> tuple[float, float, float, float, float, list, list, list, list]:
    """
    Validate model on a dataset and return various performance metrics.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for the evaluation data
        criterion: Loss function
        device: Device to run evaluation on
        input_type: Type of input to use (str or list of str)
        threshold: Classification threshold for binary predictions
        max_batches: Maximum number of batches to process (for quick validation)
        
    Returns:
        Tuple containing:
            - avg_loss: Average loss over the dataset
            - accuracy: Classification accuracy
            - f1: F1 score
            - recall: Recall score for positive class
            - pos_rate: Positive prediction rate (percentage)
            - all_preds: List of all predictions
            - all_labels: List of all ground truth labels
            - all_probs: List of all prediction probabilities
            - all_filters: List of all filter variables (mismatch)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    all_filters = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Unpack the batch
            if len(batch) == 4:
                inputs, labels, filters, _ = batch
            else:
                inputs, labels, filters = batch
            
            # Move data to device
            for key in inputs:
                inputs[key] = inputs[key].to(device)
            
            # Forward pass
            outputs = model(**inputs)

            labels = labels.to(device).float()
            filters = filters.to(device)
            
            # Calculate loss
            if output_type == "binary":
                labels_for_loss = labels.unsqueeze(1).to(device)
                loss = criterion(outputs, labels_for_loss)
            elif output_type == "multiclass":
                loss = criterion(outputs, labels)
            else:
                raise ValueError(f"Invalid output type: {output_type}")
            
            # Handle case where loss is a tensor with multiple elements
            if isinstance(loss, torch.Tensor) and loss.numel() > 1:
                loss = loss.mean()
                
            total_loss += loss.item()
            
            # Get probabilities
            if output_type == "binary":
                probs = torch.sigmoid(outputs).squeeze()
                # Make predictions using specified threshold
                preds = (probs >= threshold).int()
            elif output_type == "multiclass":
                probs = outputs             # no softmax
                preds = torch.argmax(probs, dim=1)
            else:
                raise ValueError(f"Invalid output type: {output_type}")
            
            # Store predictions and labels for metrics calculation
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_filters.extend(filters.cpu().numpy())
            
            if max_batches and batch_idx >= max_batches:
                break
    
    # Convert to numpy arrays for metric calculation
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_filters = np.array(all_filters)
    
    # Calculate average loss
    avg_loss = total_loss / len(data_loader)
    
    # Calculate accuracy
    accuracy = (all_preds == all_labels).mean()
    
    if output_type == "binary":
        # Binary classification case
        f1 = f1_score(all_labels, all_preds, pos_label=1)
        recall = recall_score(all_labels, all_preds, pos_label=1)
        pos_count = np.sum(all_preds)
    elif output_type == "multiclass":
        # Multi-class case - convert to binary (divergent vs non-divergent)
        # Class 2 is divergent, 0 and 1 are non-divergent
        binary_preds = (all_preds == 2).astype(int)
        binary_labels = (all_labels == 2).astype(int)
        f1 = f1_score(binary_labels, binary_preds, pos_label=1)
        recall = recall_score(binary_labels, binary_preds, pos_label=1)
        pos_count = np.sum(binary_preds)
    else:
        raise ValueError(f"Invalid output type: {output_type}")
    
    # Calculate positive prediction rate
    pos_rate = pos_count / len(all_labels) * 100  # as percentage

    return avg_loss, accuracy, f1, recall, pos_rate, all_preds, all_labels, all_probs, all_filters

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: Any,
    optimizer: optim.Optimizer,
    device: torch.device,
    input_type: List[str] = ["logits"], 
    output_type: str = "binary",
    num_epochs: int = 10,
    batch_size: int = 4096,
    patience: int = 1,
    valid_freq: int = 1,  # Add valid_freq parameter
    use_wandb: bool = False,
    checkpoint_dir: str = "output/checkpoints",
    output_dir: str = "output/models",
) -> tuple[nn.Module, TrainingHistory]:
    """
    Train the model with early stopping and detailed within-epoch logging.
    Save checkpoints after each epoch and select the best one at the end.

    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        input_type: Type of input to use (str or list of str)
        num_epochs: Maximum number of epochs to train for
        batch_size: Batch size for training
        patience: Number of epochs to wait for improvement before early stopping
        valid_freq: Frequency of validation checks, in times per epoch
        use_wandb: Whether to log metrics to wandb
        checkpoint_dir: Directory to save checkpoints to

    Returns:
        Trained model and training history object
    """
    # Initialize model and tracker
    model.to(device)
    
    # Initialize TrainingHistory with wandb support if requested
    history = TrainingHistory(use_wandb=use_wandb, checkpoint_dir=checkpoint_dir)
    counter = 0  # Counter for early stopping
    
    # Calculate validation frequency based on valid_freq
    val_check_freq = max(1, len(train_loader) // valid_freq)

    # Print training header
    print("\nTraining Progress:")
    print("=================")
    print(
        f"{'Epoch':^6}|{'Batch':^8}|{'Loss':^8}|{'Val Loss':^8}|{'Val Acc':^7}|{'Val Rec':^7}|{'Pos Rate':^8}"
    )
    print("-" * 62)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            # Unpack the batch
            if len(batch) == 4:
                inputs, labels, filters, _ = batch
            else:
                inputs, labels, filters = batch

            # Move data to device
            for key in inputs:
                inputs[key] = inputs[key].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(**inputs)

            labels = labels.to(device).float()

            if output_type == "binary":
                loss = criterion(outputs, labels.unsqueeze(1))
            elif output_type == "multiclass":
                loss = criterion(outputs, labels)
            else:
                raise ValueError(f"Invalid output type: {output_type}")

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Track batch statistics
            batch_loss = loss.item()
            running_loss += batch_loss
            
            # Update batch stats with batch_idx and epoch for wandb logging
            history.update_batch_stats(batch_loss, batch_idx, epoch)

            # Log batch-level progress at regular intervals
            if batch_idx % val_check_freq == 0 or batch_idx == len(train_loader) - 1:
                print_training_progress(epoch, num_epochs, batch_idx, batch_loss)

                # Perform a quick validation check at regular intervals based on val_check_freq
                if batch_idx > 0 and (batch_idx % val_check_freq == 0 or batch_idx == len(train_loader) - 1):
                    # Use validate_model with max_batches=1 for quick validation
                    quick_val_loss, quick_val_accuracy, quick_val_f1, quick_val_recall, quick_val_pos_rate, _, _, _, _ = validate_model(
                        model, val_loader, criterion, device, input_type, output_type, threshold=0.5, max_batches=1
                    )
                    val_metrics = {
                        'val_loss': quick_val_loss,
                        'val_accuracy': quick_val_accuracy,
                        'val_recall': quick_val_recall,
                        'val_pos_rate': quick_val_pos_rate
                    }
                    print_training_progress(epoch, num_epochs, batch_idx, batch_loss, val_metrics)

        # Calculate average training loss for the epoch
        epoch_train_loss = running_loss / len(train_loader)
        
        # Full validation at the end of each epoch
        val_loss, val_accuracy, val_f1, val_recall, val_pos_rate, _, _, _, _ = validate_model(
            model, val_loader, criterion, device, input_type, output_type
        )
        
        # Update epoch statistics with epoch for wandb logging
        history.update_epoch_stats(
            epoch_train_loss, val_loss, val_accuracy, val_f1, val_recall, val_pos_rate, epoch
        )
        
        # Print epoch summary
        val_metrics = {
            'train_loss': epoch_train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'val_f1': val_f1,
            'val_recall': val_recall,
            'val_pos_rate': val_pos_rate
        }
        print_training_progress(epoch, num_epochs, None, None, val_metrics, is_epoch_summary=True)
        
        # Save checkpoint and check for early stopping
        val_metrics_dict = {
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_f1": val_f1,
            "val_recall": val_recall,
            "val_pos_rate": val_pos_rate,
        }
        
        improved = history.save_checkpoint(epoch, model, optimizer, val_metrics_dict)
        
        if improved:
            counter = 0
            print(f"New best validation loss: {val_loss:.4f} at epoch {epoch+1}")
        else:
            counter += 1
            print(f"No improvement for {counter}/{patience} epochs")
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
        print("-" * 62)
    
    # Load the best model
    model = history.load_best_model(model, device)
    
    # Plot training curves
    training_curves_path = os.path.join(output_dir, "training_curves.png")
    history.plot_training_curves(training_curves_path)
    
    # Return the model and history object (not just the dictionary)
    return model, history

class InputLabelDataset(Dataset):
    """
    The dataset contains the following columns:
    - logits: The logits of the model
    - last_hidden_states: The last hidden states of the model
    - token: The token IDs 
    - label_column: The label column
    - mismatch: The mismatch label (used as a filter)

    A simplified dataset class that uses HuggingFace's set_format to efficiently convert data to tensors.
    """
    def __init__(self, dataset, input_type=["logits"], input_prefix="small_", label_column="divergent"):
        """
        Initialize the dataset with pre-processed tensors.
        
        Args:
            dataset: Hugging Face dataset
            input_type: List of input types to use (e.g., ["logits", "hidden_states", "token"])
            input_prefix: Prefix for input features (e.g., "small_")
            label_column: The column name of the label
        """
        self.input_type = input_type if isinstance(input_type, list) else [input_type]
        self.dataset = dataset
        self.label_column = label_column
        
        # Check if "mismatch" column exists, if not, create it by duplicating "divergent"
        if "mismatch" not in dataset.column_names:
            print("'mismatch' column not found. Creating it by duplicating 'divergent'.")
            self.dataset = self.dataset.map(
                create_mismatch,
                batched=False
            )
        
        # Define column names based on input_type and prefix
        self.logits_col = f"{input_prefix}logits"
        self.hidden_states_col = f"{input_prefix}last_hidden_states"
        self.token_col = f"{input_prefix}token"
        
        # Precompute which columns to include
        self.use_logits = "logits" in self.input_type and self.logits_col in dataset.column_names
        self.use_hidden_states = "hidden_states" in self.input_type and self.hidden_states_col in dataset.column_names
        self.use_token = "token" in self.input_type and self.token_col in dataset.column_names
        
        # Ensure mask column exists
        if "mask" not in dataset.column_names:
            print("'mask' column not found. Creating default mask of 1s.")
            self.dataset = self.dataset.map(
                create_mask,
                batched=False
            )

        # Set format to PyTorch tensors for required columns
        columns = [self.label_column, "mismatch", "mask"]
        if self.use_logits:
            columns.append(self.logits_col)
        if self.use_hidden_states:
            columns.append(self.hidden_states_col)
        if self.use_token:
            columns.append(self.token_col)

        # Convert input to columns
        columns = [
            self.label_column,
            "mismatch",
            "mask",
            *([self.logits_col] if self.use_logits else []),
            *([self.hidden_states_col] if self.use_hidden_states else []),
            *([self.token_col] if self.use_token else []),
        ]

        # One-time type casting so tensors are already in the correct dtype
        self.dataset = self.dataset.cast_column(self.label_column, Value("int64"))
        self.dataset = self.dataset.cast_column("mismatch", Value("int64"))
        self.dataset = self.dataset.cast_column("mask", Value("int64"))

        if self.use_logits and self.logits_col in self.dataset.column_names:
            self.dataset = self.dataset.cast_column(self.logits_col, Sequence(Value("float32")))
        if self.use_hidden_states and self.hidden_states_col in self.dataset.column_names:
            self.dataset = self.dataset.cast_column(self.hidden_states_col, Sequence(Value("float32")))
        if self.use_token and self.token_col in self.dataset.column_names:
            self.dataset = self.dataset.cast_column(self.token_col, Value("int64"))

        # Convert dataset to PyTorch tensors
        self.dataset.set_format(type="torch", columns=columns)
        
        # Print dataset info
        print(f"Dataset prepared with {len(self.dataset)} samples.")
        print(f"Using input types: {self.input_type}")
        
        # Print tensor shapes for debugging
        sample = self.dataset[0]  # a dict of torch.Tensors
        if self.use_logits:
            print(f"Logits shape: {sample[self.logits_col].shape}")
        if self.use_hidden_states:
            print(f"Hidden states shape: {sample[self.hidden_states_col].shape}")
        if self.use_token:
            print(f"Token shape: {sample[self.token_col].shape}")
            
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        row = self.dataset[idx]

        label = row[self.label_column]
        filter_var = row["mismatch"]
        mask = row["mask"]
        
        inputs = {}
        if self.use_logits:
            inputs["logits"] = row[self.logits_col]
        if self.use_hidden_states:
            inputs["hidden_states"] = row[self.hidden_states_col]
        if self.use_token:
            inputs["token"] = row[self.token_col]
        
        return inputs, label, filter_var, mask

def get_probabilities_and_labels(model, data_loader, device, input_type=["logits"], output_type="binary"):
    """
    Get model predictions and ground truth labels for a dataset.
    
    Args:
        model: The trained model
        data_loader: DataLoader for the dataset
        device: Device to run inference on
        input_type: Type of input to use (str or list of str)
        output_type: Type of output to use ("binary" or "multi-class")
        
    Returns:
        Tuple of (probabilities, labels, filters)
    """
    # TODO: uniform output format for multiclass, especially labels and probs

    # Create a dummy criterion that won't be used for actual loss calculation
    if output_type == "binary":
        dummy_criterion = nn.BCEWithLogitsLoss()
    elif output_type == "multiclass":
        dummy_criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Invalid output type: {output_type}")

    # Use validate_model to get predictions, but we'll discard most of the metrics
    avg_loss, accuracy, f1, recall, pos_rate, all_preds, all_labels, all_probs, all_filters = validate_model(
        model, data_loader, dummy_criterion, device, input_type, output_type
    )
    print(f"Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}, Positive rate: {pos_rate:.4f}")

    # If label == -1, mask should be 0
    all_filters[all_labels == -1] = 0

    return all_probs, all_labels, all_filters


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a critical classifier with optional wandb integration"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/hidden_state_sample/20250430.json",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Whether to use wandb for tracking experiments",
    )
    parser.add_argument(
        "--validate_model_path",
        type=str,
        default=None,
        help="Path to the trained model to load for validation, if given, skip training",
    )
    return parser.parse_args()


def main(config: dict, use_wandb: bool = False, validate_model_path: Optional[str] = None):
    """Main function to train and evaluate the model."""
    # Initialize wandb if requested
    if use_wandb:
        # Initialize wandb with the config
        wandb.init(
            project="FlexThink", 
            config=config,
            name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags=["classifier", config["model"]["model_type"]]
        )

    """Model"""
    model_config = config["model"]
    model = create_classifier(model_config["model_type"], **model_config["init_args"])

    # Get input type from model config
    input_type = model_config.get("input_type", ["logits"])
    print(f"Using input types: {input_type}")

    output_type = model_config.get("output_type", "binary")

    """Data"""
    data_config = config["data"]

    # load training dataset
    train_dataset_path: Union[str, List[str]] = data_config["train"]["path"]
    # Check if train_dataset_path is a list
    if isinstance(train_dataset_path, str):
        train_dataset_path = [train_dataset_path]
    # Load and concatenate datasets
    print(f"Loading and concatenating training datasets from: {train_dataset_path}")
    train_datasets = []
    for dataset_path in train_dataset_path:
        train_datasets.append(load_dataset_from_config(dataset_path))
    if len(train_datasets) > 1:
        train_dataset = concatenate_datasets(train_datasets)
    else:
        train_dataset = train_datasets[0]

    train_input_prefix = data_config["train"].get("input_prefix", "small_")
    label_column = data_config["train"].get("label_column", "divergent")
    # load test dataset
    if (not "split_test_from_train" in data_config["test"]) or (not data_config["test"]["split_test_from_train"]):
        test_dataset_path: Union[str, List[str]] = data_config["test"]["path"] # Add type hint
        # Check if test_dataset_path is a list
        if isinstance(test_dataset_path, list):
            # Load and concatenate datasets
            print(f"Loading and concatenating test datasets from: {test_dataset_path}")
            test_datasets = [load_dataset_from_config(p) for p in test_dataset_path]
            test_dataset = concatenate_datasets(test_datasets)
        else:
            # Load single dataset
            print(f"Loading test dataset from: {test_dataset_path}")
            test_dataset = load_dataset_from_config(test_dataset_path)
        test_input_prefix = data_config["test"].get("input_prefix", "small_")
    else:
        # Split train dataset into train and test
        print("Splitting training dataset into train and test sets...")
        test_size = data_config["test"].get("test_size", 0.2)
        random_seed = data_config["test"].get("random_seed", 42)

        # convert divergent to ClassType
        if "similarity_to_divergent" in data_config["train"].get("process", []):
            similarity_threshold = data_config["train"]["process"]["similarity_threshold"]
            train_dataset = train_dataset.map(
                lambda x: {"divergent": 1.0 if x["similarity"] <= similarity_threshold else 0.0},
                batched=False,
            )
        # Create divergent_class column using dataset.cast
        train_dataset = train_dataset.cast_column(
            "divergent",
            ClassLabel(num_classes=2, names=["non-critical", "critical"], id=[0,1])
        )

        # Use Hugging Face's train_test_split method
        split_datasets = train_dataset.train_test_split(
            test_size=test_size,
            seed=random_seed,
            stratify_by_column="divergent"
        )

        # Reassign train dataset and create test dataset
        train_dataset = split_datasets["train"]
        test_dataset = split_datasets["test"]
        test_input_prefix = train_input_prefix  # Use the same prefix as train dataset

        print(f"Split complete. Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

    datasets = {"train": train_dataset, "test": test_dataset}
    input_prefixes = {"train": train_input_prefix, "test": test_input_prefix}

    # process and filter datasets
    for split, dataset in datasets.items():
        # ensure divergent column exists
        for process_config in data_config[split].get("process", []):
            if process_config["type"] == "similarity_to_divergent":
                similarity_threshold = process_config["similarity_threshold"]
                datasets[split] = dataset.map(
                    lambda x: {"divergent": 1.0 if x["similarity"] <= similarity_threshold else 0.0},
                    batched=False,
                )
            if process_config["type"] == "mismatch_divergent_to_type":
                datasets[split] = dataset.map(
                    lambda x: {"type": 2 if x["divergent"] == 1 else (1 if x["mismatch"] == 1 and x["divergent"] == 0 else 0)},
                    batched=False,
                )
                datasets[split] = datasets[split].cast_column(
                    "type",
                    ClassLabel(num_classes=3, names=["non-critical", "critical", "mismatch"], id=[0,1,2])
                )
                print(f"Dataset {split} statistics:")
                print(f"  - Number of samples: {len(datasets[split])}")
                print(f"  - Number of critical samples: {sum(1 for x in datasets[split]['type'] if x == 2)}")
                print(f"  - Number of non-critical samples: {sum(1 for x in datasets[split]['type'] if x == 1)}")
                print(f"  - Number of mismatch samples: {sum(1 for x in datasets[split]['type'] if x == 0)}")

            else:
                raise ValueError(f"Invalid process type: {process_config['type']}")

        # filter data ids with too many critical samples
        for filter_config in data_config[split].get("filter", []):
            if filter_config["type"] == "filter_data_id_critical":
                datasets[split] = filter_data_id_critical(
                    datasets[split],
                    max_critical_ratio=filter_config["max_critical_ratio"],
                    min_samples_per_data_id=filter_config["min_samples_per_data_id"],
                )
            elif filter_config["type"] == "is_mismatch":
                datasets[split].set_format(
                    type="torch",
                    columns=["mismatch"]
                )
                datasets[split] = datasets[split].filter(filter_is_mismatch, batched=True, num_proc=32)
            elif filter_config["type"] == "has_divergent":
                datasets[split].set_format(
                    type="torch",
                    columns=["divergent"]
                )
                datasets[split] = datasets[split].filter(filter_has_divergent, batched=True, num_proc=32)
            elif filter_config["type"] == "downsample_match_data":
                # Convert dataset to PyTorch tensors for mismatch column
                datasets[split].set_format(
                    type="torch",
                    columns=["mismatch"]
                )

                # Get indices where mismatch is True and False
                mismatch_indices = [i for i, x in enumerate(datasets[split]["mismatch"]) if x == 1]
                match_indices = [i for i, x in enumerate(datasets[split]["mismatch"]) if x == 0]
                divergent_indices = [i for i, x in enumerate(datasets[split]["divergent"]) if x == 1]

                if filter_config["align"]=="mismatch":
                    num_mismatch_samples = len(mismatch_indices)
                    sampled_match_indices = np.random.choice(match_indices, size=num_mismatch_samples, replace=False)
                elif filter_config["align"]=="divergent":
                    num_divergent_samples = len(divergent_indices)
                    sampled_match_indices = np.random.choice(match_indices, size=num_divergent_samples, replace=False)
                else:
                    raise ValueError(f"Invalid align type: {filter_config['align']}")

                # Combine indices and sort them
                selected_indices = sorted(mismatch_indices + sampled_match_indices.tolist())

                # Select only the sampled indices
                datasets[split] = datasets[split].select(selected_indices)

                print(f"Downsampled dataset: {len(datasets[split])} samples (mismatch: {len(mismatch_indices)}, match: {len(sampled_match_indices)})")

            else:
                raise ValueError(f"Invalid filter type: {filter_config['type']}")

        # Filter out samples where mask == 0 (question tokens)
        if "mask" in datasets[split].column_names:
            before = len(datasets[split])
            # Reset format to avoid torch tensors in filter
            datasets[split].set_format(type=None)
            datasets[split] = datasets[split].filter(lambda ex: ex["mask"] == 1 and ex["divergent"] in [0, 1], num_proc=32)
            after = len(datasets[split])
            print(f"Filtered {split} dataset by mask: {before} -> {after}")

    # Use multiple workers for faster data loading if available
    num_workers = 4 if torch.cuda.is_available() else 0
    batch_size = config["training"]["params"]["batch_size"]

    # Create optimized datasets with pre-processed tensors
    train_tensor_dataset = InputLabelDataset(
        datasets["train"], 
        input_type=input_type, 
        input_prefix=input_prefixes["train"],
        label_column=label_column
    )

    test_tensor_dataset = InputLabelDataset(
        datasets["test"], 
        input_type=input_type, 
        input_prefix=input_prefixes["test"],
        label_column=label_column
    )

    # Create data loaders with the optimized datasets (no custom collate function needed)
    train_loader = DataLoader(
        train_tensor_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_tensor_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )

    """Training setup"""
    training_config = config["training"]
    device = torch.device(training_config["params"]["device"])

    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=training_config["optimizer"]["lr"], weight_decay=training_config["optimizer"]["weight_decay"])

    # Calculate class weights with higher weight for critical class
    label_column = training_config["loss"].get("label_column", "divergent")
    if training_config["loss"]["type"] == "BCEWithLogitsLoss":
        if training_config["loss"]["recall_factor"] is not None:
            print("Using class weights to balance recall")
            n_class_0 = sum(1 for x in datasets["train"][label_column] if x == 0)
            n_class_1 = sum(1 for x in datasets["train"][label_column] if x == 1)
            class_weights = torch.tensor(
                [
                    1.0,  # Non-critical
                    n_class_0
                    / n_class_1
                    * training_config["loss"]["recall_factor"],  # Higher weight for critical class to improve recall
                ],
                dtype=torch.float32,
                device=device
            )
            print(f"Using class weights: {class_weights}")
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(
                    [class_weights[1] / class_weights[0]],
                    device=device
                )
            )
        else:
            print("Using default BCEWithLogitsLoss")
            criterion = nn.BCEWithLogitsLoss()
    elif training_config["loss"]["type"] == "FocalLoss":
        criterion = FocalLoss(
            alpha=training_config["loss"]["alpha"],
            gamma=training_config["loss"]["gamma"],
        )
    elif training_config["loss"]["type"] == "DPOLoss":
        n_class_0 = sum(1 for x in datasets["train"][label_column] if x == 0)
        n_class_1 = sum(1 for x in datasets["train"][label_column] if x == 1)
        criterion = DPOLoss(
            beta=training_config["loss"]["beta"],
            positive_prob=n_class_1 / (n_class_0 + n_class_1)
        )
    elif training_config["loss"]["type"] == "CrossEntropyLoss":
        if training_config["loss"]["recall_factor"] is not None:
            print("Using class weights to improve recall")
            n_class_divergent = sum(1 for x, y in zip(datasets["train"][label_column], datasets["train"]["mismatch"]) if x == 1)
            n_class_non_divergent_mismatch = sum(1 for x, y in zip(datasets["train"][label_column], datasets["train"]["mismatch"]) if x == 0 and y == 1)
            n_class_match = sum(1 for x in datasets["train"]["mismatch"] if x == 0)

            # Calculate class weights with recall_factor applied to divergent class
            class_weights = torch.tensor(
                [
                    1.0,  # Base weight for match class
                    n_class_match / n_class_non_divergent_mismatch,  # Weight for non-divergent but mismatch class
                    n_class_match / n_class_divergent * training_config["loss"]["recall_factor"],  # Weight for divergent class
                ],
                device=device
            )
            print(f"Using class weights: {class_weights}")
            print(f"Sample counts - divergent: {n_class_divergent}, non-divergent mismatch: {n_class_non_divergent_mismatch}, match: {n_class_match}")
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            print("Using default CrossEntropyLoss")
            criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Invalid loss function: {training_config['loss']['type']}")

    print(f"Using criterion: {criterion}")

    # Train the model
    if validate_model_path is None:
        # Get valid_freq from config, default to 1 if not specified
        valid_freq = training_config["params"].get("valid_freq", 1)

        model, history = train_model(
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            device=device,
            input_type=input_type,  # Pass input_type to train_model
            output_type=output_type,
            num_epochs=training_config["params"]["num_epochs"],
            batch_size=training_config["params"]["batch_size"],
            patience=training_config["params"]["patience"],
            valid_freq=valid_freq,  # Pass valid_freq to train_model
            use_wandb=use_wandb,
            checkpoint_dir=config["output"].get("checkpoint_dir", None),
            output_dir=config["output"].get("output_dir", None),
        )
    else:
        print("Validating model only")
        # Load model using the load_model function from classifier.py
        history = TrainingHistory(use_wandb=use_wandb, checkpoint_dir=None) # dummy history is fine
        model, model_config = load_model(validate_model_path, device=device)

    """Optimize Threshold"""
    optimizing_config = config["optimizing"]

    all_probs, all_labels, all_filters = get_probabilities_and_labels(model, test_loader, device, input_type, output_type=output_type)

    # Get output directory for saving plots and data
    output_config = config["output"]
    output_dir = output_config["output_dir"]
    
    # Create output directory early to ensure it exists for all plotting functions
    os.makedirs(output_dir, exist_ok=True)
    
    is_skip_optimization = (optimizing_config["type"] == "skip")
    if not is_skip_optimization:
        pre_opt_accuracy, pre_opt_precision, pre_opt_recall, pre_opt_f1, pre_opt_positive_rate = standard_eval_pipeline(
            all_probs,
            all_labels,
            all_filters,
            output_dir = output_dir, 
            filename = "confusion_matrix_pre_opt.png")
        best_threshold, accuracy, precision, recall, f1, positive_rate, is_succeded = optimization_pipeline(
            all_probs,
            all_labels,
            all_filters,
            optimizing_config = optimizing_config,
            output_dir = output_dir
        )  
    else:
        print("Skipping threshold optimization")
        accuracy, precision, recall, f1, positive_rate = standard_eval_pipeline(all_probs, all_labels, all_filters, output_dir = output_dir)
        best_threshold = 0.5
        is_succeded = True

    # Log final results to wandb
    history.log_final_results(
        best_threshold, accuracy, precision, recall, f1, positive_rate, is_succeded
    )      

    """Output"""
    # output_dir was already defined above for optimization_pipeline and directory created
    model_name = output_config["model_name"] if output_config["model_name"] is not None else f"classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    model_path = os.path.join(output_dir, model_name) 

    # Use the save_model function to save the final model
    save_model(
        model=model,
        output_file=model_path,
        threshold=best_threshold if best_threshold is not None else 0.5,
        input_type=input_type
    )

    # Save detailed training results to config.json
    # Split directory and filename, then change suffix
    model_dir, model_filename = os.path.split(model_path)
    model_name = os.path.splitext(model_filename)[0]
    config_path = os.path.join(model_dir, model_name + ".json")
    history_dict = history.get_history_dict()
    training_results = {
        "model_path": model_path,
        "results": {
            "threshold": float(best_threshold),
            "best_epoch": history_dict.get("best_epoch"),
            "best_val_loss": float(history_dict.get("best_val_loss", 0.0)),
        }
    }
    training_results["results"]["final_metrics"] = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "positive_rate": float(positive_rate)
    }
    if not is_skip_optimization:
        training_results["results"]["pre_opt_metrics"] = {
            "accuracy": float(pre_opt_accuracy),
            "precision": float(pre_opt_precision),
            "recall": float(pre_opt_recall),
            "f1": float(pre_opt_f1),
            "positive_rate": float(pre_opt_positive_rate)
        }
    save_config = config.copy()
    save_config["result"] = training_results
    # Save the config file
    with open(config_path, 'w') as f:
        json.dump(save_config, f, indent=2)

    print(f"Training results saved to {config_path}")

    # Plot training curves using the history object's method with correct file path
    training_curves_path = os.path.join(output_dir, "training_curves.png")
    history.plot_training_curves(training_curves_path)

    # Log images and model artifact to wandb
    # history.log_model_artifact(model_path, config_path)
    history.finish_wandb_run()

    print("\nAnalysis complete. Check training_curves.png, confusion_matrix.png, and positive_rate_recall_curve_default.png for visualizations.")

if __name__ == "__main__":
    """Configuration"""
    args = parse_args()
    config_path = args.config
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(config)

    main(config, args.use_wandb, args.validate_model_path)
