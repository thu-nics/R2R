from sklearn.metrics import f1_score, precision_score, recall_score
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union, List, Dict, Any
from transformers import Trainer

from r2r.train.logger import print_evaluation_results, plot_positive_rate_recall_curve

def objective_optimal_threshold(
    pre_computed_probs,
    pre_computed_labels,
    pre_computed_filters=None,
    metric: str = "f1"
) -> float:
    """
    Find the optimal threshold for binary classification based on the specified metric.
    
    Args:
        pre_computed_probs: Pre-computed probabilities from the model
        pre_computed_labels: Pre-computed ground truth labels
        pre_computed_filters: Pre-computed filter variables (mismatch). If not given, no filtering is applied.
        metric: Metric to optimize ('f1', 'accuracy', 'recall', 'precision')
        
    Returns:
        Optimal threshold value
    """
    # Get probabilities and labels
    probs, labels = pre_computed_probs, pre_computed_labels
    
    # Apply filters if requested
    if pre_computed_filters is not None:
        # Select samples where filter is 1 (mismatched)
        mask = (pre_computed_filters == 1)
        probs = probs[mask]
        labels = labels[mask]
        print(f"Using {sum(mask)}/{len(mask)} samples after filtering.")
    
    # Try different thresholds
    thresholds = np.linspace(0.1, 0.9, 9)
    best_score = 0
    best_threshold = 0.5  # Default threshold
    
    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        if metric == 'f1':
            score = f1_score(labels, preds, pos_label=1)
        elif metric == 'accuracy':
            score = accuracy_score(labels, preds)
        elif metric == 'recall':
            score = recall_score(labels, preds, pos_label=1)
        elif metric == 'precision':
            score = precision_score(labels, preds, pos_label=1)
        else:
            raise ValueError(f"Invalid metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    print(f"Optimal threshold: {best_threshold:.2f} with {metric} score: {best_score:.4f}")
    return best_threshold

def threshold_optimal_threshold(
    pre_computed_probs,
    pre_computed_labels,
    pre_computed_filters=None,
    min_recall: float = 0.9,
    num_thresholds: int = 100
) -> float:
    """
    Find the optimal threshold that minimizes positive prediction rate while 
    maintaining a minimum recall rate.
    
    Args:
        pre_computed_probs: Pre-computed probabilities from the model
        pre_computed_labels: Ground truth labels
        pre_computed_filters: Pre-computed filter variables (mismatch)
        min_recall: Minimum recall rate to maintain (default: 0.9)
        num_thresholds: Number of threshold values to try (default: 100)
        
    Returns:
        Optimal threshold value
    """
    # Get probabilities and labels
    probs, labels = pre_computed_probs, pre_computed_labels
    
    # Apply filters if requested
    if pre_computed_filters is not None:
        # Select samples where filter is 1 (mismatched)
        mask = (pre_computed_filters == 1)
        probs = probs[mask]
        labels = labels[mask]
        print(f"Using {sum(mask)}/{len(mask)} samples after filtering.")
    
    # Try different thresholds with finer granularity
    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    
    # Track best threshold and its metrics
    best_threshold = 0.5  # Default threshold
    best_pos_rate = 100.0  # Initialize with worst case (100%)
    best_recall = 0.0
    
    # Store results for all thresholds for visualization
    results = []
    
    for threshold in thresholds:
        # Make predictions with current threshold
        preds = (probs >= threshold).astype(int)
        
        # Calculate recall
        current_recall = recall_score(labels, preds, pos_label=1)
        
        # Calculate positive prediction rate
        pos_count = np.sum(preds)
        pos_rate = pos_count / len(labels) * 100  # as percentage
        
        # Store results
        results.append((threshold, current_recall, pos_rate))
        
        # Check if this threshold meets the minimum recall requirement
        if current_recall >= min_recall:
            # If it has a lower positive rate than our current best, update
            if pos_rate < best_pos_rate:
                best_pos_rate = pos_rate
                best_threshold = threshold
                best_recall = current_recall

    is_succeded = True

    # If no threshold meets the minimum recall, find the one with highest recall
    if best_threshold == 0.5 and best_recall < min_recall:
        is_succeded = False
        print(f"Warning: Could not find threshold with minimum recall of {min_recall}.")
        print("Selecting threshold with highest recall instead.")
        
        # Sort by recall (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        best_threshold = results[0][0]
        best_recall = results[0][1]
        best_pos_rate = results[0][2]
    
    print(f"Optimal threshold: {best_threshold:.4f}")
    print(f"Metrics at optimal threshold:")
    print(f"  - Recall: {best_recall:.4f}")
    print(f"  - Positive rate: {best_pos_rate:.2f}%")
        
    return best_threshold, is_succeded

def calculate_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate recall while ignoring labels with value -1.
    
    Recall is defined as TP / (TP + FN), where:
    - TP (True Positives): Samples correctly predicted as positive (label=1, pred=1)
    - FN (False Negatives): Samples incorrectly predicted as negative (label=1, pred=0)
    
    Labels with value -1 are ignored in this calculation.
    
    Args:
        y_true: Ground truth labels (1 for positive, 0 for negative, -1 to ignore)
        y_pred: Predicted labels (1 for positive, 0 for negative)
        
    Returns:
        Recall score (between 0.0 and 1.0)
    """
    # Create mask for valid labels (not -1)
    valid_mask = (y_true != -1)
    
    # Filter out invalid labels
    filtered_true = y_true[valid_mask]
    filtered_pred = y_pred[valid_mask]
    
    # If no positive samples, return 0
    if np.sum(filtered_true == 1) == 0:
        return 0.0
    
    # Calculate recall: TP / (TP + FN)
    true_positives = np.sum((filtered_true == 1) & (filtered_pred == 1))
    actual_positives = np.sum(filtered_true == 1)
    
    return true_positives / actual_positives if actual_positives > 0 else 0.0

def standard_eval_pipeline(
    all_probs,
    all_labels,
    all_filters,
    output_dir: str = ".",
    filename="confusion_matrix.png"
):
    mask = (all_filters == 1)
    filtered_probs = all_probs[mask]
    filtered_labels = all_labels[mask]
    default_predictions = (all_probs >= 0.5).astype(int)
    filtered_predictions = default_predictions[mask]

    # First evaluate with default threshold of 0.5
    print("\n" + "=" * 60)
    print("                EVALUATION WITH THRESHOLD 0.5                     ")

    mask = (all_filters == 1)
    default_predictions = (all_probs >= 0.5).astype(int)
    accuracy, precision, recall, f1, positive_rate = print_evaluation_results(
        all_labels, default_predictions, all_probs, output_dir, filename
    )

    print("=" * 60)
    print(f"Using {sum(mask)}/{len(mask)} samples within mismatches.")
    
    print_evaluation_results(
        filtered_labels, filtered_predictions, filtered_probs, output_dir, filename
    )

    return accuracy, precision, recall, f1, positive_rate

def optimization_pipeline(
    all_probs,
    all_labels,
    all_filters,
    optimizing_config,
    output_dir: str = ".",
    filename="confusion_matrix_post_opt.png"
):  
    mask = (all_filters == 1)
    filtered_probs = all_probs[mask]
    filtered_labels = all_labels[mask]
    default_predictions = (all_probs >= 0.5).astype(int)
    filtered_predictions = default_predictions[mask]

    # Find optimal threshold
    print("\nFinding optimal threshold...")
    
    if optimizing_config["type"] == "objective":
        is_succeded = True
        best_threshold = objective_optimal_threshold(
            pre_computed_probs=all_probs,
            pre_computed_labels=all_labels,
            pre_computed_filters=None,
            metric=optimizing_config["metric"],
        )
    elif optimizing_config["type"] == "threshold":
        min_recall = optimizing_config.get("min_recall", 0.9)
        best_threshold, is_succeded = threshold_optimal_threshold(
            pre_computed_probs=all_probs,
            pre_computed_labels=all_labels,
            pre_computed_filters=None,
            min_recall=min_recall,
        )
    else:
        raise NotImplementedError

    # Final evaluation with optimal threshold
    print("\n" + "=" * 60)
    print("                EVALUATION WITH OPTIMIZED THRESHOLD                     ")

    predictions = (all_probs >= best_threshold).astype(int)
    accuracy, precision, recall, f1, positive_rate = print_evaluation_results(
        all_labels, predictions, all_probs, output_dir, filename
    )

    # Use filtered data for final evaluation
    filtered_predictions = predictions[mask]
    print("=" * 60)
    print(f"Using {sum(mask)}/{len(mask)} samples within mismatches.")
    print_evaluation_results(
        filtered_labels, filtered_predictions, filtered_probs, output_dir, filename
    )

    # Plot positive rate vs recall curve with the optimal threshold
    plot_positive_rate_recall_curve(
        y_true=all_labels,
        probabilities=all_probs,
        current_threshold=best_threshold,
        output_dir=output_dir,
    )

    return best_threshold, accuracy, precision, recall, f1, positive_rate, is_succeded