import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
import numpy as np
import pandas as pd
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
import csv
from typing import Dict, Any, Optional, Union, List
from tqdm import tqdm
from datetime import datetime
import wandb

from r2r.models.router import save_model, load_model

def create_mask(example):
    return {"mask": 1}
def create_mismatch(example):
    return {"mismatch": example["divergent"]}
def filter_data_id_critical(
    dataset: Any,
    max_critical_ratio: float = 0.5,
    min_samples_per_data_id: int = 5,
) -> Any:
    """
    Filter out data_ids that have too many critical samples.

    Args:
        dataset: The dataset to filter
        max_critical_ratio: Maximum allowed ratio of critical samples per data_id
        min_samples_per_data_id: Minimum number of samples required for a data_id to be considered

    Returns:
        Filtered dataset with problematic data_ids removed
    """
    print(f"\nFiltering data_ids with too many critical samples...")
    print(f"  - Max critical ratio: {max_critical_ratio}")
    print(f"  - Min samples per data_id: {min_samples_per_data_id}")

    # Get data_ids
    data_ids = dataset["data_id"]

    # Determine critical samples based on available columns
    critical_values = dataset["divergent"]
    is_critical = [cv == 1 for cv in critical_values]
    
    # Count critical samples per data_id
    data_id_stats = {}
    for i, (data_id, critical) in enumerate(zip(data_ids, is_critical)):
        if data_id not in data_id_stats:
            data_id_stats[data_id] = {"total": 0, "critical": 0, "indices": []}

        data_id_stats[data_id]["total"] += 1
        if critical:
            data_id_stats[data_id]["critical"] += 1
        data_id_stats[data_id]["indices"].append(i)

    # Calculate critical ratios for each data_id
    critical_ratios = []
    for data_id, stats in data_id_stats.items():
        if stats["total"] >= min_samples_per_data_id:
            critical_ratio = stats["critical"] / stats["total"]
            stats["critical_ratio"] = critical_ratio
            critical_ratios.append(critical_ratio)

    # Identify data_ids to keep
    data_ids_to_keep = []
    data_ids_to_filter = []

    for data_id, stats in data_id_stats.items():
        # Skip data_ids with too few samples
        if stats["total"] < min_samples_per_data_id:
            continue

        critical_ratio = stats["critical"] / stats["total"]

        if critical_ratio <= max_critical_ratio:
            data_ids_to_keep.append(data_id)
        else:
            data_ids_to_filter.append(data_id)

    # Get indices to keep
    indices_to_keep = []
    for data_id in data_ids_to_keep:
        indices_to_keep.extend(data_id_stats[data_id]["indices"])

    # Print statistics
    total_data_ids = len(data_id_stats)
    kept_data_ids = len(data_ids_to_keep)
    filtered_data_ids = len(data_ids_to_filter)

    print(f"\nFiltering results:")
    print(f"  - Total data_ids: {total_data_ids}")
    print(
        f"  - Kept data_ids: {kept_data_ids} ({kept_data_ids/total_data_ids*100:.2f}%)"
    )
    print(
        f"  - Filtered data_ids: {filtered_data_ids} ({filtered_data_ids/total_data_ids*100:.2f}%)"
    )

    # Original and new dataset sizes
    original_size = len(dataset)
    new_size = len(indices_to_keep)
    print(f"  - Original dataset size: {original_size}")
    print(f"  - New dataset size: {new_size} ({new_size/original_size*100:.2f}%)")

    # Print distribution of critical ratios
    print("\nDistribution of critical ratios across data_ids:")
    if critical_ratios:
        # Calculate distribution statistics
        critical_ratio_stats = {
            "min": min(critical_ratios),
            "max": max(critical_ratios),
            "mean": sum(critical_ratios) / len(critical_ratios),
            "median": sorted(critical_ratios)[len(critical_ratios) // 2],
        }

        print(f"  - Min: {critical_ratio_stats['min']:.4f}")
        print(f"  - Max: {critical_ratio_stats['max']:.4f}")
        print(f"  - Mean: {critical_ratio_stats['mean']:.4f}")
        print(f"  - Median: {critical_ratio_stats['median']:.4f}")

        # Create histogram bins
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        bin_counts = [0] * (len(bins) - 1)

        for ratio in critical_ratios:
            for i in range(len(bins) - 1):
                if bins[i] <= ratio < bins[i + 1]:
                    bin_counts[i] += 1
                    break
                # Handle the case where ratio is exactly 1.0
                elif i == len(bins) - 2 and ratio == bins[i + 1]:
                    bin_counts[i] += 1
                    break

        # Print histogram
        print("\n  Critical Ratio Histogram:")
        for i in range(len(bins) - 1):
            bin_start = bins[i]
            bin_end = bins[i + 1]
            count = bin_counts[i]
            percentage = count / len(critical_ratios) * 100
            bar = "#" * int(percentage / 2)  # Scale the bar length
            print(
                f"  {bin_start:.1f}-{bin_end:.1f}: {count:4d} ({percentage:5.1f}%) {bar}"
            )

        # List top 10 data_ids with highest critical ratios
        print("\n  Top 10 data_ids with highest critical ratios:")
        print(
            f"  {'Data ID':20} | {'Total Samples':12} | {'Critical Samples':15} | {'Critical Ratio':13}"
        )
        print(f"  {'-'*20} | {'-'*12} | {'-'*15} | {'-'*13}")

        # Sort data_ids by critical ratio
        sorted_data_ids = sorted(
            [
                (data_id, stats)
                for data_id, stats in data_id_stats.items()
                if stats["total"] >= min_samples_per_data_id
            ],
            key=lambda x: x[1]["critical"] / x[1]["total"],
            reverse=True,
        )

        # Print top 10
        for data_id, stats in sorted_data_ids[:10]:
            total = stats["total"]
            critical = stats["critical"]
            ratio = critical / total
            print(
                f"  {str(data_id)[:20]:20} | {total:12d} | {critical:15d} | {ratio:13.4f}"
            )
    else:
        print("  No data_ids with sufficient samples found.")

    # Filter the dataset
    filtered_dataset = dataset.select(indices_to_keep)

    
    original_critical = sum(1 for d in dataset["divergent"] if d == 1)
    new_critical = sum(1 for d in filtered_dataset["divergent"] if d == 1)
    
    original_critical_ratio = original_critical / original_size
    new_critical_ratio = new_critical / new_size

    print(f"\nClass distribution:")
    print(
        f"  - Original critical ratio: {original_critical_ratio:.4f} ({original_critical}/{original_size})"
    )
    print(
        f"  - New critical ratio: {new_critical_ratio:.4f} ({new_critical}/{new_size})"
    )

    return filtered_dataset

def filter_is_mismatch(
    examples: Any,
) -> Any:
    """
    Filter out samples that are not mismatches.
    """
    return examples["mismatch"] == 1

def filter_has_divergent(
    examples: Any,
) -> Any:
    """
    Filter out samples that do not have a divergent label.
    """
    return examples["divergent"] != -1

def print_training_progress(
    epoch: int,
    num_epochs: int,
    batch_idx: int,
    batch_loss: float,
    val_metrics: Optional[Dict[str, float]] = None,
    is_epoch_summary: bool = False
) -> None:
    """
    Print training progress information.
    
    Args:
        epoch: Current epoch number
        num_epochs: Total number of epochs
        batch_idx: Current batch index
        batch_loss: Loss for the current batch
        val_metrics: Optional dictionary containing validation metrics
        is_epoch_summary: Whether this is an epoch summary
    """
    if is_epoch_summary:
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"Train Loss: {val_metrics['train_loss']:.4f}, Val Loss: {val_metrics['val_loss']:.4f}")
        print(
            f"Val Accuracy: {val_metrics['val_accuracy']:.3f}, Val F1: {val_metrics['val_f1']:.3f}, Val Recall: {val_metrics['val_recall']:.3f}"
        )
        print(f"Val Positive Rate: {val_metrics['val_pos_rate']:.2f}%")
    elif val_metrics is None:
        # Print batch-level progress with fixed precision for better alignment
        print(
            f"{epoch+1:^6}|{batch_idx:^8}|{batch_loss:^8.4f}|{'-':^8}|{'-':^7}|{'-':^7}|{'-':^8}"
        )
    else:
        # Print batch-level progress with validation metrics
        print(
            f"{epoch+1:^6}|{batch_idx:^8}|{batch_loss:^8.4f}|{val_metrics['val_loss']:^8.4f}|{val_metrics['val_accuracy']:^7.3f}|{val_metrics['val_recall']:^7.3f}|{val_metrics['val_pos_rate']:^8.2f}%"
        )

def print_evaluation_results(
    y_test: Union[np.ndarray, torch.Tensor],
    predictions: Union[np.ndarray, torch.Tensor],
    probabilities: Union[np.ndarray, torch.Tensor],
    output_dir: str = ".",
    filename: str = "confusion_matrix.png"
) -> None:
    """Print detailed evaluation results with focus on recall and positive prediction rate."""
    # Print confusion matrix values - convert to numpy arrays if they're tensors
    y_test_np = y_test.numpy() if isinstance(y_test, torch.Tensor) else y_test
    predictions_np = (
        predictions.numpy() if isinstance(predictions, torch.Tensor) else predictions
    )

    tn = np.sum((y_test_np == 0) & (predictions_np == 0))
    fp = np.sum((y_test_np == 0) & (predictions_np == 1))
    fn = np.sum((y_test_np == 1) & (predictions_np == 0))
    tp = np.sum((y_test_np == 1) & (predictions_np == 1))

    total_samples = len(y_test)
    total_positives = np.sum(predictions)

    accuracy = (tp + tn) / total_samples
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)
    positive_rate = total_positives / total_samples

    # Print summary box with key metrics
    print("                     KEY PERFORMANCE METRICS                      ")
    print("=" * 60)
    print(f"RECALL (True Positive Rate): {recall*100:.2f}% (Target: > 90%)")
    print(f"POSITIVE PREDICTION RATE:    {positive_rate*100:.2f}% (Target: < 10%)")
    print(f"Note: Using 'divergent' column (1.0 = unsimilar, 0.0 = similar)")
    print("=" * 60)

    print("\nConfusion Matrix:")
    print(f"True Positives:  {tp} ({tp/(tp+fn)*100:.2f}% of actual positive cases)")
    print(f"False Negatives: {fn} ({fn/(tp+fn)*100:.2f}% of actual positive cases)")
    print(f"True Negatives:  {tn} ({tn/(tn+fp)*100:.2f}% of actual negative cases)")
    print(f"False Positives: {fp} ({fp/(tn+fp)*100:.2f}% of actual negative cases)")

    # Print additional metrics
    print(
        f"\nPositive predictions: {total_positives} out of {total_samples} ({positive_rate:.2f}%)"
    )
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"F1 Score: {f1*100:.2f}%")

    # Print probability distribution for divergent cases
    divergent_probs = probabilities[y_test == 1]
    print("\nPrediction distribution for actual divergent cases:")
    print(f"Min: {divergent_probs.min():.1f}")
    print(f"Max: {divergent_probs.max():.1f}")
    print(f"Mean: {divergent_probs.mean():.1f}")
    print(f"Median: {np.median(divergent_probs):.1f}")

    # Create and save confusion matrix visualization
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    confusion_matrix_path = os.path.join(output_dir, filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(confusion_matrix_path, bbox_inches="tight", dpi=300)
    plt.close()

    return accuracy, precision, recall, f1, positive_rate

def plot_positive_rate_recall_curve(
    y_true: Union[np.ndarray, torch.Tensor],
    probabilities: Union[np.ndarray, torch.Tensor],
    current_threshold: Optional[float] = None,
    output_dir: str = "."
) -> None:
    """
    Plot the positive prediction rate vs recall curve with varied thresholds.

    Args:
        y_true: True binary labels
        probabilities: Predicted probabilities
        current_threshold: Current threshold to highlight on the curve (optional)
        output_dir: Directory to save the output files
    """
    # Calculate metrics at different thresholds
    thresholds = np.linspace(0.01, 0.99, 100)
    recalls = []
    positive_rates = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(float)

        # Calculate recall
        recall = recall_score(y_true, predictions, pos_label=1)
        recalls.append(recall)

        # Calculate positive prediction rate
        total_positives = np.sum(predictions)
        positive_rate = total_positives / len(y_true) * 100
        positive_rates.append(positive_rate)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(positive_rates, recalls, "b-", linewidth=2)

    # Add grid for better readability
    plt.grid(True, linestyle="--", alpha=0.7)

    # Highlight the current threshold if provided
    if current_threshold is not None:
        current_predictions = (probabilities >= current_threshold).astype(float)
        current_recall = recall_score(y_true, current_predictions, pos_label=1)
        current_positive_rate = np.sum(current_predictions) / len(y_true) * 100

        plt.plot(
            current_positive_rate,
            current_recall,
            "ro",
            markersize=10,
            label=f"Current threshold: {current_threshold:.3f}",
        )

        # Add annotation for the current threshold
        plt.annotate(
            f"Threshold: {current_threshold:.3f}\nRecall: {current_recall:.3f}\nPositive Rate: {current_positive_rate:.2f}%",
            xy=(current_positive_rate, current_recall),
            xytext=(current_positive_rate + 5, current_recall - 0.05),
            arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.8),
        )

    # Add labels and title
    plt.xlabel("Positive Prediction Rate (%)")
    plt.ylabel("Recall")

    # Add information about the dataset type in the title
    plt.title(
        f"Positive Rate vs Recall Curve"
    )

    # Add legend if current threshold is provided
    if current_threshold is not None:
        plt.legend(loc="best")

    # Save the figure
    plt.tight_layout()
    png_file_path = os.path.join(output_dir, "positive_rate_recall_curve.png")
    plt.savefig(png_file_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(
        f"Positive Rate vs Recall curve saved as '{png_file_path}'"
    )

    # Save data to CSV using pandas DataFrame
    df_data = pd.DataFrame({
        'Threshold': thresholds,
        'Positive Rate': positive_rates,
        'Recall': recalls
    })
    csv_file_path = os.path.join(output_dir, "positive_rate_recall_data.csv")
    df_data.to_csv(csv_file_path, index=False)
    
    print(f"Positive Rate vs Recall data saved as '{csv_file_path}'")


def plot_training_curves(history: Dict[str, Any], output_file: str = "training_curves.png") -> None:
    """
    Plot detailed training curves from a history dictionary.
    This is a wrapper around the TrainingHistory.plot_training_curves method
    for compatibility with existing code.
    
    Args:
        history: Dictionary containing training history
        output_file: Path to save the plot
    """
    # Create a temporary TrainingHistory object to use its plotting method
    temp_history = TrainingHistory()
    
    # Populate the history object with data from the dictionary
    temp_history.train_losses = history["train_loss"]
    temp_history.val_losses = history["val_loss"]
    temp_history.val_accuracies = history["val_accuracy"]
    temp_history.val_f1s = history["val_f1"]
    temp_history.val_recalls = history["val_recall"]
    temp_history.val_pos_rates = history["val_pos_rate"]
    temp_history.batch_losses = history["batch_losses"]
    temp_history.batch_steps = history["batch_steps"]
    temp_history.best_epoch = history.get("best_epoch")
    
    # Use the TrainingHistory plotting method
    temp_history.plot_training_curves(output_file)


class TrainingHistory:
    """
    Class to track and manage training history and checkpoints.
    Optionally integrates with wandb for experiment tracking.
    """
    def __init__(self, use_wandb: bool = False, checkpoint_dir: Optional[str] = None):
        """
        Initialize the history tracker with empty containers.
        
        Args:
            use_wandb: Whether to log metrics to wandb
            checkpoint_dir: Directory to save checkpoints to, if None, use "output/checkpoints"
        """
        # History tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1s = []
        self.val_recalls = []
        self.val_pos_rates = []
        
        # Batch-level tracking
        self.batch_losses = []
        self.batch_steps = []
        self.total_steps = 0
        
        # Checkpoint tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.best_epoch = None
        self.checkpoint_dir = checkpoint_dir if checkpoint_dir is not None else "output/checkpoints"
        self.checkpoints = {}
        
        # Wandb integration
        self.use_wandb = use_wandb
        
        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def update_batch_stats(self, batch_loss: float, batch_idx: int = None, epoch: int = None) -> None:
        """
        Update batch-level statistics and log to wandb if enabled.
        
        Args:
            batch_loss: Loss value for the current batch
            batch_idx: Current batch index (optional, for wandb logging)
            epoch: Current epoch (optional, for wandb logging)
        """
        self.total_steps += 1
        self.batch_losses.append(batch_loss)
        self.batch_steps.append(self.total_steps)
        
        # Log to wandb if enabled
        if self.use_wandb and batch_idx is not None and epoch is not None:
            import wandb
            wandb.log({
                "batch": batch_idx + epoch * self.total_steps,
                "batch_loss": batch_loss,
            })
    
    def update_epoch_stats(self, 
                          train_loss: float, 
                          val_loss: float, 
                          val_accuracy: float, 
                          val_f1: float, 
                          val_recall: float, 
                          val_pos_rate: float,
                          epoch: int = None) -> None:
        """
        Update epoch-level statistics and log to wandb if enabled.
        
        Args:
            train_loss: Training loss for the epoch
            val_loss: Validation loss for the epoch
            val_accuracy: Validation accuracy for the epoch
            val_f1: Validation F1 score for the epoch
            val_recall: Validation recall for the epoch
            val_pos_rate: Validation positive rate for the epoch
            epoch: Current epoch (optional, for wandb logging)
        """
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_accuracy)
        self.val_f1s.append(val_f1)
        self.val_recalls.append(val_recall)
        self.val_pos_rates.append(val_pos_rate)
        
        # Log to wandb if enabled
        if self.use_wandb and epoch is not None:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_f1": val_f1,
                "val_recall": val_recall,
                "val_pos_rate": val_pos_rate
            })
    
    def save_checkpoint(self, epoch: int, model: nn.Module, optimizer: optim.Optimizer, val_metrics: dict) -> bool:
        """
        Save a checkpoint for the given epoch and log to wandb if enabled.
        Returns True if this is the new best model.
        
        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer to save
            val_metrics: Validation metrics to save
            
        Returns:
            Whether this is the new best model
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        
        # Save model using the save_model function
        try:
            # Save optimizer state separately since save_model doesn't handle it
            optimizer_state = copy.deepcopy(optimizer.state_dict())
            
            # Save the model with its configuration and threshold
            save_model(
                model=model,
                output_file=checkpoint_path,
                threshold=0.5,  # Default threshold
                epoch=epoch + 1,
                optimizer_state_dict=optimizer_state,
                val_metrics=val_metrics
            )
            
            # Store checkpoint info
            self.checkpoints[epoch + 1] = {"path": checkpoint_path, "val_loss": val_metrics["val_loss"]}
            
            # Check if this is the best model so far
            is_best = val_metrics["val_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["val_loss"]
                self.best_model_state = copy.deepcopy(model.state_dict())
                self.best_epoch = epoch + 1
                
                # Log to wandb if enabled and this is the best model
                if self.use_wandb:
                    import wandb
                    wandb.log({
                        "best_epoch": epoch + 1,
                        "best_val_loss": val_metrics["val_loss"],
                        "best_val_accuracy": val_metrics["val_accuracy"],
                        "best_val_f1": val_metrics["val_f1"],
                        "best_val_recall": val_metrics["val_recall"],
                        "best_val_pos_rate": val_metrics["val_pos_rate"]
                    })
                
                return True  # Indicates improvement
            
        except Exception as e:
            print(f"Warning: Failed to save checkpoint: {e}")
            # Even if saving fails, still track the best model in memory
            self.checkpoints[epoch + 1] = {"path": None, "val_loss": val_metrics["val_loss"]}
            
            # Check if this is the best model so far
            is_best = val_metrics["val_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["val_loss"]
                self.best_model_state = copy.deepcopy(model.state_dict())
                self.best_epoch = epoch + 1
                
                # Log to wandb if enabled and this is the best model
                if self.use_wandb:
                    import wandb
                    wandb.log({
                        "best_epoch": epoch + 1,
                        "best_val_loss": val_metrics["val_loss"],
                        "best_val_accuracy": val_metrics["val_accuracy"],
                        "best_val_f1": val_metrics["val_f1"],
                        "best_val_recall": val_metrics["val_recall"],
                        "best_val_pos_rate": val_metrics["val_pos_rate"]
                    })
                
                return True  # Indicates improvement
        
        return False  # Indicates no improvement
    
    def load_best_model(self, model: nn.Module, device: torch.device) -> nn.Module:
        """Load the best model based on validation loss."""
        if not self.checkpoints:
            print("No checkpoints available to load")
            return model
            
        # Find the checkpoint with the lowest validation loss
        best_epoch = min(self.checkpoints.keys(), key=lambda k: self.checkpoints[k]["val_loss"])
        best_checkpoint_path = self.checkpoints[best_epoch]["path"]
        best_val_loss = self.checkpoints[best_epoch]["val_loss"]
        
        print(f"\nBest checkpoint was from epoch {best_epoch} with validation loss: {best_val_loss:.4f}")
        
        # Check if we have a saved checkpoint file for the best epoch
        if best_checkpoint_path is not None:
            print(f"Loading best checkpoint from {best_checkpoint_path}")
            
            try:
                # Use the load_model function to load the model
                loaded_model, config = load_model(best_checkpoint_path, device=device)
                
                # Transfer the loaded model's state to our model
                model.load_state_dict(loaded_model.state_dict())
                
                print(f"Successfully loaded checkpoint from epoch {best_epoch}")
                
                # Return any additional info from the config if needed
                if "threshold" in config.get("common_args", {}):
                    print(f"Loaded threshold: {config['common_args']['threshold']}")
                
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Falling back to best model state from memory")
                # Fall back to the best model state that was saved in memory during training
                if self.best_model_state is not None:
                    model.load_state_dict(self.best_model_state)
                    print(f"Loaded best model state from memory with validation loss: {best_val_loss:.4f}")
                else:
                    print(
                        "Warning: Could not load best model state. Using current model state."
                    )
        else:
            print("No saved checkpoint file available for the best epoch")
            # Use the best model state that was saved in memory during training
            if self.best_model_state is not None:
                model.load_state_dict(self.best_model_state)
                print(f"Loaded best model state from memory with validation loss: {best_val_loss:.4f}")
            else:
                print("Warning: Could not load best model state. Using current model state.")
        
        return model
    
    def plot_training_curves(self, output_file: str = "training_curves.png") -> None:
        """
        Plot detailed training curves and log to wandb if enabled.
        
        Args:
            output_file: Path to save the plot
        """
        plt.figure(figsize=(15, 10))
        
        # Plot batch-level loss
        plt.subplot(2, 2, 1)
        plt.plot(self.batch_steps, self.batch_losses)
        plt.title("Batch-level Training Loss")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        
        # Plot epoch-level metrics
        plt.subplot(2, 2, 2)
        plt.plot(self.train_losses, label="Training Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        if self.best_epoch:
            plt.axvline(
                x=self.best_epoch - 1, color="r", linestyle="--", label=f"Best Epoch ({self.best_epoch})"
            )
        plt.title("Epoch-level Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.plot(np.array(self.val_accuracies) * 100, label="Accuracy(%)")
        plt.plot(np.array(self.val_recalls) * 100, label="Recall(%)", linestyle="--", color="red")
        plt.plot(self.val_pos_rates, label="Pos Rate(%)", linestyle="-.", color="green")
        if self.best_epoch:
            plt.axvline(
                x=self.best_epoch - 1, color="r", linestyle="--", label=f"Best Epoch ({self.best_epoch})"
            )
        plt.title("Validation Metrics")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.plot(self.val_f1s)
        if self.best_epoch:
            plt.axvline(
                x=self.best_epoch - 1, color="r", linestyle="--", label=f"Best Epoch ({self.best_epoch})"
            )
        plt.title("Validation F1 Score")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.legend()
        
        plt.tight_layout()

        dirpath = os.path.dirname(output_file)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        plt.savefig(output_file)

        plt.close()
        
        print(f"Training curves saved to {output_file}")
    
    def get_history_dict(self) -> Dict[str, Any]:
        """Return the history as a dictionary."""
        return {
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
            "val_accuracy": self.val_accuracies,
            "val_f1": self.val_f1s,
            "val_recall": self.val_recalls,
            "val_pos_rate": self.val_pos_rates,
            "batch_losses": self.batch_losses,
            "batch_steps": self.batch_steps,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
        }
        
    def log_final_results(self, 
                         threshold: float, 
                         accuracy: float, 
                         precision: float, 
                         recall: float, 
                         f1: float, 
                         positive_rate: float,
                         is_succeded: bool) -> None:
        """
        Log final evaluation results to wandb.
        
        Args:
            threshold: Optimal threshold
            accuracy: Final accuracy
            precision: Final precision
            recall: Final recall
            f1: Final F1 score
            positive_rate: Final positive rate
        """
        if self.use_wandb:
            import wandb
            wandb.log({
                "final_threshold": threshold,
                "final_accuracy": accuracy,
                "final_precision": precision,
                "final_recall": recall,
                "final_f1": f1,
                "final_positive_rate": positive_rate,
                "final_is_succeded": is_succeded
            })
    
    def log_model_artifact(self, model_path: str, config_path: str) -> None:
        """
        Log the model as a wandb artifact.
        
        Args:
            model_path: Path to the saved model
            config_path: Path to the saved config
        """
        if self.use_wandb:
            import wandb
            model_artifact = wandb.Artifact(
                name=f"model-{wandb.run.id}", 
                type="model",
                description=f"Trained model with best validation loss: {self.best_val_loss:.4f}"
            )
            model_artifact.add_file(model_path)
            model_artifact.add_file(config_path)
            wandb.log_artifact(model_artifact)
    
    def log_image(self, name: str, image_path: str) -> None:
        """
        Log an image to wandb.
        
        Args:
            name: Name of the image
            image_path: Path to the image
        """
        if self.use_wandb:
            import wandb
            wandb.log({
                name: wandb.Image(image_path)
            })
    
    def finish_wandb_run(self) -> None:
        """Finish the wandb run."""
        if self.use_wandb:
            import wandb
            wandb.finish()