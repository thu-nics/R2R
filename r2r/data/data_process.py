import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import logging
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from r2r.utils.config import TOKEN_TYPE, MODEL_DICT

logger = logging.getLogger(__name__)

@dataclass
class MismatchPoint:
    """Represents a point where small (1.5B) and reference (32B) model predictions differ"""
    data_id: int
    token_id: int  # token ID in the sequence
    real_token: int
    prev_real_token: int # the previous real token of current token
    pred_small_token: int
    pred_reference_token: int
    real_text: str  # decoded text of real token
    pred_small_text: str  # decoded text of small model prediction
    pred_reference_text: str  # decoded text of reference model prediction
    current_token_samples: List[int] = None  # will store list of token samples, default to [current_token]
    update_context_tokens: List[int] = None  # will store context from last mismatch
    update_context_text: str = None  # decoded text of context from last mismatch
    context_tokens: List[int] = None  # will store context from the start of the data item to the current token
    context_text: str = None  # decoded text of context from the start of the data item to the current token

    def print(self):
        """Print mismatch information in a formatted way"""
        print(f"  Data ID: {self.data_id}")
        print(f"  Token ID: {self.token_id}")
        print('%r' % f"  Update context:{self.update_context_text}'")
        print('%r' % f"  Real token: '{self.real_text}' (token value: {self.real_token})")
        print('%r' % f"  Small model prediction: '{self.pred_small_text}' (token value: {self.pred_small_token})")
        print('%r' % f"  Reference model prediction: '{self.pred_reference_text}' (token value: {self.pred_reference_token})")

@dataclass
class DataContext:
    """Stores context information for a data item"""
    data_id: int
    start_row: int
    end_row: int
    real_tokens: List[int]  # The actual token values
    token_ids: List[int]  # The positions of tokens in sequence
    row_ids: List[int]  # Added to track original row IDs

class DataProcessor:
    def __init__(self, df: pd.DataFrame, max_tokens: int = 8192, comparison_model: str = 'reference', is_multi_pred: bool = False):
        self.data = df
        self.max_tokens = max_tokens
        self.comparison_model = comparison_model
        self.is_multi_pred = is_multi_pred

        self.eos_token = MODEL_DICT["special_tokens"]['think_end']

        self._validate_columns()
        self.data_contexts = {}  # cache for data contexts
        # Initialize both tokenizers
        small_model_path = MODEL_DICT["quick"]['model_path']
        reference_model_path = MODEL_DICT["reference"]['model_path']
        self.tokenizer_small = AutoTokenizer.from_pretrained(small_model_path)
        if self.comparison_model == 'reference':
            self.tokenizer_reference = AutoTokenizer.from_pretrained(reference_model_path)
        else:
            self.tokenizer_reference = AutoTokenizer.from_pretrained(small_model_path)
        
        logger.info(f"DataProcessor initialized with comparison_model: {comparison_model}")

    def _validate_columns(self):
        """Verify required columns exist in the CSV"""
        if self.comparison_model == 'reference':
            required_columns = [
                'row_id',
                'data_id',
                'real_token',
                'token_id',
                'token_type',
                'SLM_predictions',
                'LLM_predictions',
            ]
        else: # 'real'
            required_columns = [
                'row_id',
                'data_id',
                'real_token',
                'token_id',
                'token_type',
                'SLM_predictions',
            ]
            
        if self.is_multi_pred:
            required_columns.append('SLM_prediction_samples')
            
        missing_cols = [col for col in required_columns if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
    def _parse_prediction_samples(self, predictions_str: str) -> List[int]:
        """Parse the prediction samples from the string format [token1, token2, token3, token4]"""
        try:
            # Remove brackets and split by comma
            tokens_str = predictions_str.strip('[]').split(',')
            # Convert to integers
            return [int(token.strip()) for token in tokens_str]
        except Exception as e:
            logger.error(f"Error parsing prediction samples {predictions_str}: {str(e)}")
            return []

    def _decode_token(self, token_id: int, is_reference: bool = False) -> str:
        """Decode a single token ID to text using the appropriate tokenizer"""
        try:
            tokenizer = self.tokenizer_reference if is_reference else self.tokenizer_small
            return tokenizer.decode([token_id])
        except Exception as e:
            logger.error(f"Error decoding token {token_id}: {str(e)}")
            return f"<error_decoding_{token_id}>"
            
    def _decode_token_list(self, token_ids: List[int]) -> str:
        """Decode a list of token IDs to text"""
        try:
            # Use small model tokenizer for context since it's from real tokens
            return self.tokenizer_small.decode(token_ids)
        except Exception as e:
            logger.error(f"Error decoding tokens {token_ids}: {str(e)}")
            return f"<error_decoding_{token_ids}>"

    def find_mismatches(self) -> List[MismatchPoint]:
        """
        Find points where small model predictions differ from either:
        - reference model predictions (if comparison_model is 'reference')
        - real tokens (if comparison_model is 'real')
        Only considers tokens that are reasoning (1) or response (2) tokens
        """
        # Group by data_id first to handle the instruction boundary
        grouped_data = self.data.groupby('data_id')
        
        mismatches = []
        for data_id, data_item in tqdm(grouped_data, desc="Finding mismatches"):
            try:
                # Check if data item length exceeds max_tokens
                if len(data_item) > self.max_tokens:
                    logger.warning(f"Skipping data item {data_id} as it exceeds max token length ({len(data_item)} > {self.max_tokens})")
                    continue
                    
                # find first non-instruction token (10x faster than pandas)
                first_reasoning_idx = np.argmax(data_item['token_type'].values > TOKEN_TYPE.INPUT_INSTRUCTION)
                valid_rows = data_item.iloc[first_reasoning_idx:]
                
                if len(valid_rows) == 0:
                    logger.warning(f"No reasoning or response tokens found in data item {data_id}")
                    continue
                
                if len(valid_rows) == len(data_item):
                    logger.warning(f"No reasoning or response tokens found in data item {data_id}")
                    continue

                # Reset index to make shifting easier
                valid_rows = valid_rows.reset_index(drop=True)
                
                # Find mismatches based on comparison_model
                if self.comparison_model == 'reference':
                    # Compare small model predictions with reference model predictions
                    mismatch_mask = (valid_rows['SLM_predictions'] != valid_rows['LLM_predictions']) & (valid_rows['real_token'] != self.eos_token)
                else:  # 'real'
                    # Compare small model predictions with real tokens
                    # We need to shift because predictions are for the next token
                    # Exclude the last row since there's no next row to compare with
                    if len(valid_rows) > 1:
                        # Get small model predictions (except last row)
                        predictions = valid_rows['SLM_predictions'].iloc[:-1].astype(int).reset_index(drop=True)
                        
                        # Get real tokens from next position (skip first row)
                        next_tokens = valid_rows['real_token'].iloc[1:].astype(int).reset_index(drop=True)
                        
                        # Create mask where predictions don't match real tokens and real token isn't EOS
                        mismatch_mask = (predictions != next_tokens) & (next_tokens != self.eos_token)
                        
                        # Pad with False for the last row to keep original length
                        mismatch_mask = pd.Series(list(mismatch_mask) + [False])
                    else:
                        mismatch_mask = pd.Series([False])
                # We need to look at predictions from current row but real token from next row
                for idx in range(len(valid_rows)-1):  # -1 because we need next row
                    if not mismatch_mask[idx]:
                        continue
                        
                    current_row = valid_rows.iloc[idx]
                    next_row = valid_rows.iloc[idx + 1]
                    
                    # If multi-prediction mode is enabled, parse the prediction samples
                    if self.is_multi_pred and 'SLM_prediction_samples' in current_row:
                        token_samples = self._parse_prediction_samples(current_row['SLM_prediction_samples'])
                    else:
                        token_samples = [int(current_row['SLM_predictions'])]
                    
                    # The token_id and real_text should come from the next position
                    # since predictions are for the next token
                    if self.comparison_model == 'reference':
                        mismatch = MismatchPoint(
                            data_id=int(current_row['data_id']),
                            token_id=int(next_row['token_id']),  # Position being predicted
                            prev_real_token=int(current_row['real_token']),  # Previous real token of current token
                            real_token=int(next_row['real_token']),  # Actual next token
                            pred_small_token=int(current_row['SLM_predictions']),  # Small model's prediction for next token
                            pred_reference_token=int(current_row['LLM_predictions']),  # Reference model's prediction for next token
                            pred_small_text=self._decode_token(int(current_row['SLM_predictions']), is_reference=False),
                            pred_reference_text=self._decode_token(int(current_row['LLM_predictions']), is_reference=True),
                            real_text=self._decode_token(int(next_row['real_token'])),
                            current_token_samples=token_samples,
                            update_context_tokens=[],
                            update_context_text="",
                            context_tokens=[],
                            context_text=""
                        )
                        mismatches.append(mismatch)
                    else: # 'real'
                        mismatch = MismatchPoint(
                            data_id=int(current_row['data_id']),
                            token_id=int(next_row['token_id']),  # Position being predicted
                            prev_real_token=int(current_row['real_token']),  # Previous real token of current token
                            real_token=int(next_row['real_token']),  # Actual next token
                            pred_small_token=int(current_row['SLM_predictions']),  # Small model's prediction for next token
                            pred_reference_token = None,
                            pred_small_text=self._decode_token(int(current_row['SLM_predictions']), is_reference=False),
                            pred_reference_text=None,
                            real_text=self._decode_token(int(next_row['real_token'])),
                            current_token_samples=token_samples,
                            update_context_tokens=[],
                            update_context_text="",
                            context_tokens=[],
                            context_text=""
                        )
                        mismatches.append(mismatch)
            except Exception as e:
                logger.error(f"Error processing data item {data_id}: {str(e)}")
                continue
        
        logger.info(f"Found {len(mismatches)} total mismatches where small and {self.comparison_model} tokens differ")
        
        return mismatches

    def get_data_context(self, data_id: int) -> DataContext:
        """Get full context for a data item"""
        if data_id in self.data_contexts:
            return self.data_contexts[data_id]
        
        data_item = self.data[self.data['data_id'] == data_id].sort_values('row_id')
        context = DataContext(
            data_id=int(data_id),
            start_row=int(data_item['row_id'].min()),
            end_row=int(data_item['row_id'].max()),
            real_tokens=[int(x) for x in data_item['real_token'].tolist()],
            token_ids=[int(x) for x in data_item['token_id'].tolist()],
            row_ids=[int(x) for x in data_item['row_id'].tolist()]
        )
        self.data_contexts[data_id] = context
        
        logger.debug(f"Created context for data item {data_id} with {len(context.token_ids)} tokens")
        return context

    def extract_mismatch_context(self, mismatch: MismatchPoint, prev_mismatch: Optional[MismatchPoint] = None) -> MismatchPoint:
        """Extract context tokens from last mismatch (or data item start) to current mismatch"""
        context = self.get_data_context(mismatch.data_id)
        
        # Get index in the  context for the position being predicted
        current_idx = context.token_ids.index(mismatch.token_id)
        
        # For first mismatch in data_item, get all tokens up to current (exclusive of predicted position)
        if prev_mismatch is None:
            # Context from start of data_item up to but not including predicted position
            mismatch.context_tokens = context.real_tokens[:current_idx]
            mismatch.context_text = self._decode_token_list([int(x) for x in context.real_tokens[:current_idx]])
            mismatch.update_context_tokens = mismatch.context_tokens
            mismatch.update_context_text = mismatch.context_text
        else:
            # Get index of previous mismatch's predicted position
            prev_idx = context.token_ids.index(prev_mismatch.token_id)
            
            # Context from start of data_item up to current token
            mismatch.context_tokens = context.real_tokens[:current_idx]  # Up to but not including predicted position
            mismatch.context_text = self._decode_token_list([int(x) for x in context.real_tokens[:current_idx]])
            
            # Update context from last mismatch's position
            if prev_idx < current_idx - 1:  # Only if there are tokens between mismatches
                mismatch.update_context_tokens = context.real_tokens[prev_idx:current_idx]  # Include token at prev_idx
                mismatch.update_context_text = self._decode_token_list([int(x) for x in context.real_tokens[prev_idx:current_idx]])
        
        return mismatch

    def group_mismatches_by_data_id(self) -> Dict[int, List[MismatchPoint]]:
        """Group mismatches by data_id and extract context for each"""
        mismatches = self.find_mismatches()
        grouped = defaultdict(list)
        
        # First group by data item
        for mismatch in mismatches:
            grouped[mismatch.data_id].append(mismatch)
        
        # Sort within each data item and extract context
        for data_id, data_mismatches in tqdm(grouped.items(), desc="Finding context for mismatches"):
            data_mismatches.sort(key=lambda x: x.token_id)
            logger.info(f"Processing {len(data_mismatches)} mismatches in data item {data_id}")
            
            # Extract context for each mismatch
            prev_mismatch = None
            for mismatch in data_mismatches:
                self.extract_mismatch_context(mismatch, prev_mismatch)
                prev_mismatch = mismatch
        
        return dict(grouped)

    def save_mismatches(self, grouped_mismatches: Dict[int, List[MismatchPoint]], output_path: str):
        """Save mismatches to a JSON file"""
        # Collect statistics
        total_mismatches = sum(len(mismatches) for mismatches in grouped_mismatches.values())
        total_data_items = len(grouped_mismatches)
        avg_mismatches = total_mismatches / total_data_items if total_data_items > 0 else 0
        
        logger.info(f"Statistics:")
        logger.info(f"Total mismatches: {total_mismatches}")
        logger.info(f"Total data items with mismatches: {total_data_items}")
        logger.info(f"Average mismatches per data item: {avg_mismatches:.2f}")
        
        # Convert to JSON-serializable format
        output_data = {}
        for data_id, data_mismatches in grouped_mismatches.items():
            output_data[str(data_id)] = []
            for mismatch in data_mismatches:
                mismatch_data = {
                    'data_id': mismatch.data_id,
                    'token_id': mismatch.token_id,
                    'real_token': mismatch.real_token,
                    'pred_small_token': mismatch.pred_small_token,
                    'pred_reference_token': mismatch.pred_reference_token,
                    'pred_small_text': mismatch.pred_small_text,
                    'pred_reference_text': mismatch.pred_reference_text,
                    'real_text': mismatch.real_text,
                    'update_context_tokens': mismatch.update_context_tokens,
                    'update_context_text': mismatch.update_context_text,
                    'context_tokens': mismatch.context_tokens,
                    'context_text': mismatch.context_text
                }
                output_data[str(data_id)].append(mismatch_data)
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Saved {total_mismatches} mismatches to {output_path}")