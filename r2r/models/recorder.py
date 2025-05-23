import pandas as pd
from typing import Optional, Dict, List, Union
from dataclasses import dataclass
from collections import defaultdict
from transformers import PreTrainedTokenizer
from r2r.utils.config import QUICK_COLOR, REFERENCE_COLOR, UNDERLINE, RESET

@dataclass
class GenerationRecord:
    token_id: int
    token_str: str
    source_model: str  # 'quick' or 'reference'
    position: int
    param_size: float
    batch_id: int = 0  # Added to track which batch this record belongs to
    # Optional fields for recording mode
    quick_model_prediction: Optional[int] = None
    quick_model_entropy: Optional[float] = None
    aleatoric_uncertainty: Optional[float] = None
    epistemic_uncertainty: Optional[float] = None
    reference_model_prediction: Optional[int] = None

class GenerationRecorder:
    def __init__(self):
        self.records: List[GenerationRecord] = []
        
    def add_record(self, record: GenerationRecord):
        self.records.append(record)
    
    def get_statistics(self) -> Dict[str, Union[int, float]]:
        total_tokens = len(self.records)
        model_counts = defaultdict(int)
        agreement_count = 0
        quick_source_agreement = 0
        quick_source_total = 0
        total_params = 0
        total_aleatoric = 0
        total_epistemic = 0
        
        for record in self.records:
            model_counts[record.source_model] += 1
            total_params += record.param_size
            
            # Safely handle optional uncertainty values
            if record.aleatoric_uncertainty is not None:
                total_aleatoric += record.aleatoric_uncertainty
            if record.epistemic_uncertainty is not None:
                total_epistemic += record.epistemic_uncertainty
            
            # Check if both predictions exist before comparing
            if (record.reference_model_prediction is not None and 
                record.quick_model_prediction is not None and 
                record.quick_model_prediction == record.reference_model_prediction):
                agreement_count += 1
                if record.source_model == "quick":
                    quick_source_agreement += 1
            
            if record.source_model == "quick":
                quick_source_total += 1
        
        # Count tokens with valid uncertainty measurements
        valid_aleatoric_count = sum(1 for r in self.records if r.aleatoric_uncertainty is not None)
        valid_epistemic_count = sum(1 for r in self.records if r.epistemic_uncertainty is not None)
        
        stats = {
            "total_tokens": total_tokens,
            "quick_model_tokens": model_counts["quick"],
            "reference_model_tokens": model_counts["reference"],
            "quick_model_percentage": model_counts["quick"] / total_tokens * 100 if total_tokens > 0 else 0,
            "reference_model_percentage": model_counts["reference"] / total_tokens * 100 if total_tokens > 0 else 0,
            "model_agreement_count": agreement_count,
            "model_agreement_percentage": agreement_count / total_tokens * 100 if total_tokens > 0 else 0,
            "quick_source_agreement_count": quick_source_agreement,
            "quick_source_total": quick_source_total,
            "quick_source_agreement_percentage": quick_source_agreement / quick_source_total * 100 if quick_source_total > 0 else 0,
            "total_params_billions": total_params,
            "avg_params_billions": total_params / total_tokens if total_tokens > 0 else 0,
            "avg_aleatoric_uncertainty": total_aleatoric / valid_aleatoric_count if valid_aleatoric_count > 0 else 0,
            "avg_epistemic_uncertainty": total_epistemic / valid_epistemic_count if valid_epistemic_count > 0 else 0
        }
        return stats
    
    def get_batch_statistics(self, batch_id: int) -> Dict[str, Union[int, float]]:
        """Get statistics for a specific batch.
        
        Args:
            batch_id: The batch ID to get statistics for
            
        Returns:
            Dictionary of statistics for the specified batch
        """
        # Filter records for this batch
        batch_records = [r for r in self.records if r.batch_id == batch_id]
        
        total_tokens = len(batch_records)
        model_counts = defaultdict(int)
        agreement_count = 0
        quick_source_agreement = 0
        quick_source_total = 0
        total_params = 0
        total_aleatoric = 0
        total_epistemic = 0
        
        for record in batch_records:
            model_counts[record.source_model] += 1
            total_params += record.param_size
            
            # Safely handle optional uncertainty values
            if record.aleatoric_uncertainty is not None:
                total_aleatoric += record.aleatoric_uncertainty
            if record.epistemic_uncertainty is not None:
                total_epistemic += record.epistemic_uncertainty
            
            # Check if both predictions exist before comparing
            if (record.reference_model_prediction is not None and 
                record.quick_model_prediction is not None and 
                record.quick_model_prediction == record.reference_model_prediction):
                agreement_count += 1
                if record.source_model == "quick":
                    quick_source_agreement += 1
            
            if record.source_model == "quick":
                quick_source_total += 1
        
        # Count tokens with valid uncertainty measurements
        valid_aleatoric_count = sum(1 for r in batch_records if r.aleatoric_uncertainty is not None)
        valid_epistemic_count = sum(1 for r in batch_records if r.epistemic_uncertainty is not None)
        
        stats = {
            "batch_id": batch_id,
            "total_tokens": total_tokens,
            "quick_model_tokens": model_counts["quick"],
            "reference_model_tokens": model_counts["reference"],
            "quick_model_percentage": model_counts["quick"] / total_tokens * 100 if total_tokens > 0 else 0,
            "reference_model_percentage": model_counts["reference"] / total_tokens * 100 if total_tokens > 0 else 0,
            "model_agreement_count": agreement_count,
            "model_agreement_percentage": agreement_count / total_tokens * 100 if total_tokens > 0 else 0,
            "quick_source_agreement_count": quick_source_agreement,
            "quick_source_total": quick_source_total,
            "quick_source_agreement_percentage": quick_source_agreement / quick_source_total * 100 if quick_source_total > 0 else 0,
            "total_params_billions": total_params,
            "avg_params_billions": total_params / total_tokens if total_tokens > 0 else 0,
            "avg_aleatoric_uncertainty": total_aleatoric / valid_aleatoric_count if valid_aleatoric_count > 0 else 0,
            "avg_epistemic_uncertainty": total_epistemic / valid_epistemic_count if valid_epistemic_count > 0 else 0
        }
        return stats
    
    def get_batch_records(self, batch_id: int) -> List[GenerationRecord]:
        """Get all records for a specific batch.
        
        Args:
            batch_id: The batch ID to get records for
            
        Returns:
            List of GenerationRecord objects for the specified batch
        """
        return [r for r in self.records if r.batch_id == batch_id]
    
    def to_batch_dataframe(self, batch_id: int) -> pd.DataFrame:
        """Convert records for a specific batch to a pandas DataFrame.
        
        Args:
            batch_id: The batch ID to get records for
            
        Returns:
            DataFrame containing records for the specified batch
        """
        batch_records = self.get_batch_records(batch_id)
        return pd.DataFrame([vars(record) for record in batch_records])
    
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([vars(record) for record in self.records])
    
    def get_colored_text(self, tokenizer: PreTrainedTokenizer, show_alternatives: bool = True) -> str:
        """Returns the generated text with color coding for different models and alternative predictions.
        
        Args:
            tokenizer: The tokenizer to decode token IDs
            show_alternatives: Whether to show alternative predictions on a second line
        """
        lines = [""]  # Main text lines
        alt_lines = [""] if show_alternatives else []  # Alternative prediction lines
        current_line = 0
        current_alt_position = 0
        
        for record in self.records:
            token_str = record.token_str
            is_agreement = (record.reference_model_prediction is not None and 
                          record.quick_model_prediction == record.reference_model_prediction)
            
            # Handle newlines in token
            splits = token_str.split('\n')
            for i, split in enumerate(splits):
                if i > 0:
                    # Start new lines for both main and alt text
                    lines.append("")
                    if show_alternatives:
                        alt_lines.append("")
                    current_line += 1
                    current_alt_position = 0
                
                # Add spaces to alt_line to align with main line
                if show_alternatives:
                    while len(alt_lines[current_line]) < current_alt_position:
                        alt_lines[current_line] += " "
                
                # Add appropriate color based on source model
                if record.source_model == "quick":
                    lines[current_line] += QUICK_COLOR
                else:
                    lines[current_line] += REFERENCE_COLOR
                
                # If models disagree, show both predictions
                if not is_agreement:
                    lines[current_line] += UNDERLINE
                    lines[current_line] += split + RESET
                    
                    # Add alternative prediction to second line if enabled
                    if show_alternatives:
                        alt_token = None
                        if record.source_model == "quick":
                            # If quick model was used, show reference prediction
                            alt_token = record.reference_model_prediction
                        else:
                            # If reference model was used, show quick prediction
                            alt_token = record.quick_model_prediction
                        
                        if alt_token is not None and i == 0:  # Only show alternative on first line of split
                            alt_token_str = tokenizer.decode([alt_token])
                            alt_lines[current_line] += f"{REFERENCE_COLOR if record.source_model == 'quick' else QUICK_COLOR}↑{alt_token_str}{RESET}"
                else:
                    lines[current_line] += split + RESET
                
                current_alt_position = len(lines[current_line].replace(QUICK_COLOR, "").replace(REFERENCE_COLOR, "").replace(UNDERLINE, "").replace(RESET, ""))
        
        # Combine main and alt lines
        result = []
        if show_alternatives:
            for main_line, alt_line in zip(lines, alt_lines):
                result.append(main_line)
                if alt_line.strip():  # Only add alt line if it contains something
                    result.append(alt_line)
        else:
            result = lines
        
        return "\n".join(result)
    
    def get_batch_colored_text(self, batch_id: int, tokenizer: PreTrainedTokenizer, show_alternatives: bool = True) -> str:
        """Returns the generated text for a specific batch with color coding for different models.
        
        Args:
            batch_id: The batch ID to get the colored text for
            tokenizer: The tokenizer to decode token IDs
            show_alternatives: Whether to show alternative predictions on a second line
        
        Returns:
            Color-coded text for the specified batch
        """
        # Filter records for this batch
        batch_records = self.get_batch_records(batch_id)
        
        # Create a temporary recorder with only the batch records
        temp_recorder = GenerationRecorder()
        for record in batch_records:
            temp_recorder.add_record(record)
        
        # Use the standard method to get colored text
        return temp_recorder.get_colored_text(tokenizer, show_alternatives)
        
    def get_confusion_matrix(self) -> Dict[str, int]:
        """
        Calculate confusion matrix for model selection.
        
        Label=1 means models disagree (quick model prediction != reference model prediction)
        Label=0 means models agree (quick model prediction == reference model prediction)
        Pred=1 means reference model was used
        Pred=0 means quick model was used
        
        Returns:
            Dictionary with confusion matrix values:
            - true_positive: Label=1, Pred=1 (correctly used reference model when needed)
            - false_positive: Label=0, Pred=1 (unnecessarily used reference model)
            - false_negative: Label=1, Pred=0 (incorrectly used quick model when reference was needed)
            - true_negative: Label=0, Pred=0 (correctly used quick model when it was sufficient)
        """
        true_positive = 0  # Label=1, Pred=1 (correctly used reference model when needed)
        false_positive = 0  # Label=0, Pred=1 (unnecessarily used reference model)
        false_negative = 0  # Label=1, Pred=0 (incorrectly used quick model when reference was needed)
        true_negative = 0  # Label=0, Pred=0 (correctly used quick model when it was sufficient)
        
        for record in self.records:
            # Skip records without both predictions (should not happen in is_record=True mode)
            if record.quick_model_prediction is None or record.reference_model_prediction is None:
                continue
                
            # Determine the actual label (whether models disagree)
            label = 1 if record.quick_model_prediction != record.reference_model_prediction else 0
            
            # Determine the prediction (which model was used)
            pred = 1 if record.source_model == "reference" else 0
            
            # Update confusion matrix
            if label == 1 and pred == 1:
                true_positive += 1
            elif label == 0 and pred == 1:
                false_positive += 1
            elif label == 1 and pred == 0:
                false_negative += 1
            elif label == 0 and pred == 0:
                true_negative += 1
        
        return {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative
        }
    
    def print_confusion_matrix(self) -> None:
        """
        Print the confusion matrix in a readable format.
        
        Label=1 means models disagree (quick model prediction != reference model prediction)
        Label=0 means models agree (quick model prediction == reference model prediction)
        Pred=1 means reference model was used
        Pred=0 means quick model was used
        """
        cm = self.get_confusion_matrix()
        
        # Calculate metrics
        total = sum(cm.values())
        accuracy = (cm["true_positive"] + cm["true_negative"]) / total if total > 0 else 0
        precision = cm["true_positive"] / (cm["true_positive"] + cm["false_positive"]) if (cm["true_positive"] + cm["false_positive"]) > 0 else 0
        recall = cm["true_positive"] / (cm["true_positive"] + cm["false_negative"]) if (cm["true_positive"] + cm["false_negative"]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        print("                  | Actual Label |")
        print("                  | Disagree (1) | Agree (0) |")
        print("------------------|--------------|-----------|")
        print(f"Predicted | Ref (1) | {cm['true_positive']:^12} | {cm['false_positive']:^9} |")
        print(f"Model     | Quick(0)| {cm['false_negative']:^12} | {cm['true_negative']:^9} |")
        print("------------------|--------------|-----------|")
        
        # Print metrics
        print("\nMetrics:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        
        # Print interpretation
        print("\nInterpretation:")
        print(f"- True Positives (TP): {cm['true_positive']} - Correctly used reference model when models disagree")
        print(f"- False Positives (FP): {cm['false_positive']} - Unnecessarily used reference model when models agree")
        print(f"- False Negatives (FN): {cm['false_negative']} - Incorrectly used quick model when models disagree")
        print(f"- True Negatives (TN): {cm['true_negative']} - Correctly used quick model when models agree")
