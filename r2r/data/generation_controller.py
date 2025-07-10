import re
import torch
from torch import Tensor
# Remove HuggingFace imports
import logging
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer
import json
import numpy as np
from dataclasses import dataclass
from itertools import chain
import sys
import os
import jieba
# Add sglang import and get_tokenizer
from r2r.utils.config import MODEL_DICT
import sglang as sgl
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
from r2r.data.utils.convert_eos_tokens import save_semantic_tokens_config
from r2r.data.data_process import MismatchPoint

# Add the parent directory to Python path before any other imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from r2r.utils.sampling import sample_token

from nltk.tokenize import sent_tokenize
from nltk.tokenize import PunktSentenceTokenizer

logger = logging.getLogger(__name__)

@dataclass
class DivergePoint:
    """Represents a divergence point with context and continuation data"""
    # Identifiers
    data_id: int
    token_id: int
    
    # Prediction data
    pred_small_token: List[int]
    pred_small_text: str
    
    # Divergence data
    small_diverge_text: str
    reference_diverge_text: str
    common_context: str
    
    def print(self):
        """Print divergence point information in a formatted way"""
        print('%r' % f"Refer diverge text: {self.reference_diverge_text}")
        print('%r' % f"Small diverge text: {self.small_diverge_text}")


class ModelController:
    def __init__(self, comparison_model: str = 'reference', mem_fraction_static: float = 0.5, tp_size: int = 1, dp_size: int = 1, base_gpu_id_main: int = 0, base_gpu_id_reference: int = 1, disable_cuda_graph: bool = False):
        """Initialize models for generation control using sglang with token-based processing"""
        # Load models
        logger.info("Loading models...")
        small_model_path = MODEL_DICT["continuation_main"]['model_path']
        if comparison_model == 'reference':
            reference_model_path = MODEL_DICT['reference']['model_path']

        # Initialize HuggingFace tokenizer directly
        self.tokenizer = AutoTokenizer.from_pretrained(small_model_path, trust_remote_code=True)

        # Initialize sglang models using Engine with skip_tokenizer_init=True
        self.main_model = sgl.Engine(
            model_path=small_model_path, 
            dtype="bfloat16", 
            mem_fraction_static=mem_fraction_static,
            skip_tokenizer_init=True,
            tp_size=tp_size,
            dp_size=dp_size,
            enable_custom_logit_processor=True,
            base_gpu_id=base_gpu_id_main,
            disable_cuda_graph=disable_cuda_graph
        )

        # Only load reference model if needed
        self.comparison_model = comparison_model

        # Use the main model if both models are the same
        self.reuse_main_model = (comparison_model == 'reference') and (small_model_path == reference_model_path)

        if comparison_model == 'reference':
            if self.reuse_main_model:
                print("Reusing main model for reference model")
                self.reference_model = self.main_model
            else:
                self.reference_model = sgl.Engine(
                    model_path=reference_model_path, 
                    dtype="bfloat16", 
                    mem_fraction_static=mem_fraction_static,
                    skip_tokenizer_init=True,
                    tp_size=tp_size,
                    dp_size=dp_size,
                    enable_custom_logit_processor=True,
                    base_gpu_id=base_gpu_id_reference,
                    disable_cuda_graph=disable_cuda_graph
                )

        # Custom EOS tokens - keep the same logic for tracking EOS tokens
        eos_tokens_config_path = os.path.join(os.path.dirname(__file__), 'eos_tokens_config.json')
        if not os.path.exists(eos_tokens_config_path):
            tokenizer = AutoTokenizer.from_pretrained(small_model_path)
            save_semantic_tokens_config(tokenizer, eos_tokens_config_path)
            print(f"Missing eos_tokens_config.json, saved it with {small_model_path} tokenizer to {eos_tokens_config_path}")
        
        with open(eos_tokens_config_path, 'r') as f:
            eos_tokens_config = json.load(f)
            if eos_tokens_config["tokenizer_name"] != MODEL_DICT["quick"]["model_name"]:
                tokenizer = AutoTokenizer.from_pretrained(small_model_path)
                save_semantic_tokens_config(tokenizer, eos_tokens_config_path, tokenizer_name=MODEL_DICT["quick"]["model_name"])
                print(f"Missing eos_tokens_config.json, saved it with {small_model_path} tokenizer to {eos_tokens_config_path}")
            else:
                print(f"Found eos_tokens_config.json at {eos_tokens_config_path}")

        with open(eos_tokens_config_path, 'r') as f:
            eos_tokens_config = json.load(f)
            self.stop_token_ids = [int(key) for key in eos_tokens_config["semantic_tokens_mapping"].keys()]
            self.stop_token_ids.append(MODEL_DICT["special_tokens"]["think_start"])
            self.stop_token_ids.append(MODEL_DICT["special_tokens"]["think_end"])
            
        # self.eos_token_ids = [
        #     151648, # <think>
        #     151649, # </think>
        #     151643, # 
        #     4710, # '\n\n
        #     18611,15514,6320,83900,70191,94521,24796,32057,53589,50524,52324,33933,45806,2146,44364,26469, # ??\n\n
        #     271, 2219, 1837, 26487, 66426, 15441, 3876, 44611, 692, 21518, 41025, 3554, 10452, 382, 8680, 1447, 401, 68327, 1339, 1939, 19347, 2533, 18797, 19324, 4257, 43738, 630, 58629, 60543, 78988,75048,3407,26850,17701, # ?\n\n
        #     198, 4894, 698, 4956, 25046, 13744, 1248, 1006, 340, 5618, 16930, 345, 6913, 624, 5894, 510, 280, 14750, 397, 5267, 62799, 921, 12924, 3989, 515, 7360, 532, 88141,8997,41453,94432, # ?\n
        #     22614,33947,50347,7129,89262,11248,5661,14288,39071,79295,5929,1171,1019,33084,51377,84274,10952,7088, #??\n
        #     13,1773,6313,11319,30,1112 # end of data item
        # ]
        
        self.eos_tokens = [
            self.tokenizer.decode([token_id]) for token_id in self.stop_token_ids
        ]

        # Initialize the DeterministicLogitProcessor
        self.deterministic_logit_processor = DeterministicLogitProcessor(
            stop_token_ids=self.stop_token_ids,
            eos_token_id=self.tokenizer.eos_token_id,
            num_continuation=-1  # Default value, will be updated when used
        )

        logger.info("ModelController initialized successfully")

    def _is_eos_generated(self, token_id: int) -> bool:
        """Check if generated token is an EOS token"""
        return token_id in self.stop_token_ids

    @staticmethod
    def _greedy_generate(
        model: sgl.Runtime,
        prompt: str,
        max_new_tokens: Optional[int] = 128,
        temperature: Optional[float] = 0.0,
        eos_token_ids: Optional[list] = None,
        top_p: float = 1.0,
        num_continuation: int = 1
    ) -> List[int]:
        """
        Custom greedy generation with sglang.
        
        Args:
            model: The sglang Runtime instance to use for generation
            prompt: Text prompt to generate from
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for softmax (higher = more random, lower = more deterministic)
            eos_token_ids: List of token IDs that should trigger early stopping
            top_p: Top-p probability threshold for nucleus sampling (0 < top_p ≤ 1)
            num_continuation: Number of EOS tokens to encounter before stopping generation
        Returns:
            List of generated token IDs
        """

        # Define a custom logit processor function to track EOS tokens
        class EOSTracker:
            def __init__(self, eos_token_ids, num_continuation):
                self.eos_token_ids = eos_token_ids if eos_token_ids else []
                self.num_continuation = num_continuation
                self.eos_count = 0

            def __call__(self, input_id, logits):
                # Check if the last token was an EOS token
                if input_id in self.eos_token_ids:
                    self.eos_count += 1
                    if self.eos_count >= self.num_continuation:
                        # Signal generation to stop
                        return True, None
                return False, None

        eos_tracker = EOSTracker(eos_token_ids, num_continuation)

        # Create an sglang state with custom parameters
        with sgl.interpretation(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            logit_processor=eos_tracker if eos_token_ids else None
        ):
            # Define a simple generation task
            @sgl.function
            def generate(s):
                s += sgl.gen(prompt=prompt)
                return s.generated_text

        # Execute the generation
        result = generate()

        # Convert the generated text to token IDs
        tokenizer = sgl.Tokenizer(model=model.model_name_or_path)
        generated_tokens = tokenizer.encode(result)

        return generated_tokens

    def generate_continuation(
        self,
        update_context_tokens: List[List[int]],
        current_token: List[int],
        model_type: str,
        past_key_values: Optional[Tuple] = None,  # Not needed with sglang but kept for API compatibility
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        num_samples: int = 1,
        top_p: float = 1.0,
        top_k: int = -1,
        num_continuation: int = 1
    ) -> List[Dict]:
        """
        Generate continuation from given context using sglang batch inference with token IDs.
        
        Args:
            update_context_tokens: List of lists of tokens from last mismatch to current position
            current_token: List of tokens to generate the continuation for
            model_type: Either 'small' or 'reference'
            past_key_values: Not used in sglang implementation
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for generation (higher = more random, lower = more deterministic)
            num_samples: Number of samples to generate per input
            top_p: Top-p probability threshold for nucleus sampling (0 < top_p ≤ 1)
            num_continuation: Number of continuations to generate
            
        Returns:
            List of dictionaries, each containing:
                - generated_tokens: List of generated token IDs`
                - past_key_values: None (not used in sglang)
                - generated_text: Decoded text of extracted tokens (for compatibility)
        """
        # Select appropriate model for model-based generation
        model = self.main_model if model_type == 'small' else self.reference_model

        # Process inputs for batch generation
        input_ids_list = []

        for ctx_tokens, curr_tok in zip(update_context_tokens, current_token):
            # Create the input by appending current token to context tokens
            input_ids = ctx_tokens + [curr_tok]
            # If num_samples > 1, duplicate each input
            for _ in range(num_samples):
                input_ids_list.append(input_ids[:])

        # Prepare sampling parameters
        sampling_params = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_new_tokens": max_new_tokens,
        }

        # Set EOS token behavior based on num_continuation value
        if num_continuation == 1:
            # For single continuation, use our custom EOS tokens
            sampling_params["stop_token_ids"] = self.stop_token_ids
        elif num_continuation > 1:
            # For multiple continuations, update the logit processor's num_continuation
            self.deterministic_logit_processor.num_continuation = num_continuation
            sampling_params["custom_params"] = {'dummy_for_req': 1}
        # When num_continuation == -1, we don't set stop_token_ids, 
        # letting the model generate until its default EOS or max_new_tokens

        # Generate completions in batch using token IDs directly
        try:
            if num_continuation > 1:
                outputs = model.generate(input_ids=input_ids_list, sampling_params=sampling_params, custom_logit_processor=self.deterministic_logit_processor.to_str())
            else:
                outputs = model.generate(input_ids=input_ids_list, sampling_params=sampling_params)

            # Process results
            results = []
            for output in outputs:
                # Get the generated token ids (output_ids contains only the new tokens)
                generated_token_ids = output["output_ids"]

                # Remove the last EOS token if it exists
                if generated_token_ids[-1] == self.tokenizer.eos_token_id:
                    generated_token_ids = generated_token_ids[:-1]

                # Decode to text for compatibility with existing code
                generated_text = self.tokenizer.decode(generated_token_ids)
                results.append({
                    'generated_tokens': generated_token_ids,
                    'past_key_values': None,  # Not used in sglang Engine
                    'generated_text': generated_text
                })
            return results

        except Exception as e:
            logger.error(f"Error during batch generation: {str(e)}")
            # Return empty results on error
            return [{
                'generated_tokens': [],
                'past_key_values': None,
                'generated_text': ''
            } for _ in range(len(input_ids_list))]

    def extract_real_continuation(
        self,
        full_real_tokens: List[int],
        current_token_index: int,
        max_new_tokens: int = 128,
        num_continuation: int = 1
    ) -> Dict:
        """
        Extract actual continuation from real tokens instead of generating with a model.
        
        Args:
            full_real_tokens: Complete list of real tokens from the data
            current_token_index: Index of the current token in full_real_tokens
            max_new_tokens: Maximum number of tokens to extract
            num_continuation: Number of continuations/sentences to extract, -1 for unlimited
        Returns:
            Dictionary containing:
                - generated_tokens: List of extracted real token IDs
                - generated_text: Decoded text of extracted tokens (for compatibility)
        """
        # Calculate the end index (limited by sequence length)
        end_index = min(current_token_index + max_new_tokens + 1, len(full_real_tokens))

        # Get the 'generated' tokens (which are actually the real next tokens)
        generated_tokens = full_real_tokens[current_token_index+1:end_index]

        # Track EOS tokens to limit to num_continuation continuations, but only if num_continuation > 0
        if num_continuation > 0:
            eos_count = 0
            for i, token in enumerate(generated_tokens):
                if token in self.stop_token_ids:
                    eos_count += 1
                    if eos_count >= num_continuation:
                        generated_tokens = generated_tokens[:i+1]
                        break

        # Handle the case where generated_tokens is empty (at end of sequence)
        if not generated_tokens:
            # If we're at the end of the sequence, set generated_tokens to contain the EOS token
            generated_tokens = [self.tokenizer.eos_token_id]
        else:
            # Remove the last EOS token if it exists
            if generated_tokens[-1] == self.tokenizer.eos_token_id:
                generated_tokens = generated_tokens[:-1]

        # Decode the tokens to text for compatibility
        generated_text = self.tokenizer.decode(generated_tokens)

        return {
            'generated_tokens': generated_tokens,
            'generated_text': generated_text
        }

    def decode_tokens(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        try:
            return self.tokenizer.decode(token_ids)
        except Exception as e:
            logger.error(f"Error decoding tokens: {str(e)}")
            return f"<error_decoding_{token_ids}>"

    def sent_tokenize_with_format(self, text):
        tokenizer = PunktSentenceTokenizer()
        spans = list(tokenizer.span_tokenize(text))
        sentences = []

        for i, (start, end) in enumerate(spans):
            # 向后扩展末尾空白符，包括 \n、\r、空格、制表符等
            extended_end = end
            while extended_end < len(text) and text[extended_end] in ' \t\n\r':
                extended_end += 1
            sentences.append(text[start:extended_end])
        
        return sentences

    def custom_sent_tokenize(self, text: str) -> List[Tuple[str, int, int]]:
        def split_chinese_sentences(text):
            """use jieba to split chinese sentences, keep all original characters"""
            # 将换行符替换为特殊标记
            text = text.replace('\n', '<NEWLINE>')

            # 使用jieba的分句功能
            sentences = []
            current_sentence = ''

            # 定义标点符号
            punctuation = '。！？；…'

            for i, char in enumerate(text):
                current_sentence += char
                # 如果当前字符是标点符号，或者到达文本末尾
                if char in punctuation or i == len(text) - 1:
                    # 处理当前句子
                    if current_sentence.strip():
                        # 恢复换行符
                        current_sentence = current_sentence.replace('<NEWLINE>', '\n')
                        sentences.append(current_sentence.strip())
                    current_sentence = ''

            # 处理最后一个句子（如果没有以标点符号结尾）
            if current_sentence.strip():
                current_sentence = current_sentence.replace('<NEWLINE>', '\n')
                sentences.append(current_sentence.strip())

            return sentences

        def contains_chinese(text):
            """check if text contains chinese characters"""
            return any('\u4e00' <= char <= '\u9fff' for char in text)

        def split_by_newlines(text: str) -> List[str]:
            """Split text by newlines while preserving them, prioritizing \n and \n\n splits"""
            if not text:
                return []

            result = []
            segments = []
            current = []
            i = 0
            
            while i < len(text):
                if text[i] == '\n':
                    # Check for double newline
                    if i + 1 < len(text) and text[i + 1] == '\n':
                        # Add current text if exists
                        if current:
                            segments.append(''.join(current))
                            current = []
                        # Add the double newline
                        segments.append('\n\n')
                        i += 2
                    else:
                        # Add current text if exists
                        if current:
                            segments.append(''.join(current))
                            current = []
                        # Add the single newline
                        segments.append('\n')
                        i += 1
                else:
                    current.append(text[i])
                    i += 1
            
            # Add any remaining text
            if current:
                segments.append(''.join(current))
            
            # Merge segments that don't end with newlines with the following segment
            i = 0
            while i < len(segments):
                if i + 1 < len(segments) and not segments[i].endswith('\n'):
                    result.append(segments[i] + segments[i + 1])
                    i += 2
                else:
                    result.append(segments[i])
                    i += 1
            
            return [s for s in result if s]

        # First get initial sentence splits
        sentences = self.sent_tokenize_with_format(text)

        # Process each sentence and split by newlines
        processed_sentences = []
        for sentence in sentences:
            if contains_chinese(sentence):
                # For Chinese text, first split into sentences then split by newlines
                chinese_sentences = split_chinese_sentences(sentence)
                for chinese_sent in chinese_sentences:
                    processed_sentences.extend(split_by_newlines(chinese_sent))
            else:
                # For non-Chinese text, directly split by newlines
                processed_sentences.extend(split_by_newlines(sentence))

        # Calculate positions for each segment
        sentence_positions = []
        start_idx = 0

        for sentence in processed_sentences:
            # Find the sentence in text starting from start_idx
            start_idx = text.find(sentence, start_idx)
            if start_idx == -1:  # Handle case where sentence is not found
                continue

            end_idx = start_idx + len(sentence) - 1  # -1 because end_idx should point to last character

            sentence_positions.append((sentence, start_idx, end_idx))
            start_idx = end_idx + 1  # Move start_idx to next position

        return sentence_positions

    def get_latest_context(
        self,
        context_text: str,
        pred_text: str,
        model_output_text: str,
        is_next_context: bool,
        num_continuation: int = 1,
        previous_context: int = 0,
        common_previous_context: int = -1,
        is_skip_input: bool = True,
    ) -> Tuple[str, str]:
        """
        Extract the sentence containing the unique occurrence of pred_text, plus optional previous and subsequent sentences.
        If previous_context is -1, return the full text.
        If common_previous_context is -1, return all previous text, otherwise return specified number of previous sentences.

        Args:
            context_text: The context text before the generated content
            pred_text: The text to locate in the context
            model_output_text: The text generated by the model
            is_next_context: Whether to look after the pred_text (used when pred_text is an EOS token)
            num_continuation: Number of continuation sentences to include after the current sentence
            previous_context: Number of previous sentences to include before the current sentence
            common_previous_context: Number of previous sentences to include in all_previous (-1 for all)
            skip_input: Whether to skip the input text, which is everything before <think>
        Returns:
            Tuple containing:
                - all_previous: All text before the result (controlled by common_previous_context)
                - result: String containing the combined context (previous sentences + current sentence + continuation sentences)
        """
        # Handle special case with <think> tags in context_text
        if "<think>" in context_text and is_skip_input:
            context_text = context_text.split("<think>")[-1]

        # Concatenate all text components
        full_text = context_text + pred_text + model_output_text

        if previous_context == -1:
            return "", full_text

        # Calculate the index of pred_text in the full text
        if pred_text[0] == " ":
            pred_text_index = len(context_text) + 1
        else:
            pred_text_index = len(context_text)

        # Adjust index for EOS token case
        if is_next_context:
            pred_text_index += len(pred_text)

        # Tokenize the full text into sentences with position information
        sentence_positions = self.custom_sent_tokenize(full_text)

        # Find the sentence containing the pred_text
        current_sentence_idx = -1
        for idx, (sentence, start_idx, end_idx) in enumerate(sentence_positions):
            if start_idx <= pred_text_index <= end_idx:
                current_sentence_idx = idx
                break

        # If we didn't find the sentence, return empty string
        if current_sentence_idx == -1:
            return "", "EMPTY"

        # Calculate the range of sentences to include in result
        start_idx = max(0, current_sentence_idx - previous_context)

        # For continuations: either include up to num_continuation sentences after current,
        # or include all remaining sentences if fewer than requested
        if num_continuation == 1:
            end_idx = current_sentence_idx + 1  # Just the current sentence
        else:
            end_idx = min(
                len(sentence_positions), current_sentence_idx + num_continuation
            )

        # Get all previous text based on common_previous_context
        all_previous = ""
        if start_idx > 0:
            # Determine how many previous sentences to include
            if common_previous_context == -1:
                prev_start = 0  # Include all previous sentences
            else:
                prev_start = max(0, start_idx - common_previous_context)  # Include specified number of sentences
            
            # Combine the previous sentences
            for idx in range(prev_start, start_idx):
                all_previous += sentence_positions[idx][0]

        # Combine the sentences for result
        result = ""
        for idx in range(start_idx, end_idx):
            result += sentence_positions[idx][0]

        return all_previous, result

    def shutdown(self):
        """Shut down the Engine instances to free resources"""
        try:
            self.main_model.shutdown()
            if self.comparison_model == 'reference':
                self.reference_model.shutdown()
            logger.info("ModelController engines shut down successfully")
        except Exception as e:
            logger.error(f"Error shutting down engines: {str(e)}")

class DeterministicLogitProcessor(CustomLogitProcessor):
    """A dummy logit processor that changes the logits to always
    sample the given token id.
    """
    def __init__(self, stop_token_ids: List[int], eos_token_id: int, num_continuation: int):
        self.eos_token_id = eos_token_id
        self.stop_token_ids_set = set(stop_token_ids)  # Convert to set for O(1) lookup
        self.num_continuation = num_continuation

    def __call__(self, logits: Tensor, custom_param_list: List[Dict]):
        """
        Input:
            logits: shape (batch_size, vocab_size)
            custom_param_list: List[Dict]. The size of the list is the same as the batch size.
                Each element in the list is a dictionary with the keys and values: input_ids (List[int]), output_ids (List[int]), __req__ (Req); as well as the keys and values specified in the custom_params in sampling_params.
        """
        if self.num_continuation == -1:
            raise ValueError("num_continuation should not be -1. Ensure it is set before invoking the model's generation function, which requires a valid continuation limit.")
            
        batch_size = logits.shape[0]
        assert batch_size == len(custom_param_list)
        
        # Vectorized argmax computation for all batch items at once
        predicted_tokens = logits.argmax(dim=-1)  # shape: (batch_size,)
        
        # Initialize mask for items that need EOS forcing
        eos_mask = torch.zeros(batch_size, dtype=torch.bool, device=logits.device)
        
        for i, param_dict in enumerate(custom_param_list):
            req = param_dict["__req__"]
            if not hasattr(req, "num_continue_count"):
                req.num_continue_count = 0
            
            # Check if this batch item has reached the continuation limit
            if req.num_continue_count >= self.num_continuation:
                eos_mask[i] = True
            
            # Use set lookup for O(1) time complexity instead of O(n) list search
            if predicted_tokens[i].item() in self.stop_token_ids_set:
                req.num_continue_count += 1
        
        # Apply EOS forcing to all relevant batch items at once using vectorized operations
        if eos_mask.any():
            logits[eos_mask, :] = -float("inf")
            logits[eos_mask, self.eos_token_id] = 1.0
       
        return logits