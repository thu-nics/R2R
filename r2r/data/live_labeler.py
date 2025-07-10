from typing import List, Tuple, Optional, Union

from r2r.data.generation_controller import ModelController, DivergePoint
from r2r.data.verify_model import VerifyModel, ComparisonPoint
from transformers import AutoTokenizer

from r2r.utils.config import MODEL_DICT, TOKEN


class LiveDivergentLabeler:
    """
    A class that handles live labeling of divergent tokens between small and reference models,
    and verifies the quality of continuations for divergent points.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        # verify config
        verify_model_name: str = 'Qwen/Qwen2.5-1.5B-Instruct',
        verify_mode: str = 'divergent',
        # continuation config
        continuation_max_new_tokens: int = 128,
        verifier_max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        num_samples: int = 1,
        num_continuation: int = 1,
        previous_context: int = 0
    ):
        """
        Initialize the live divergent labeler.
        
        Args:
            verify_model_name: Name of the model to use for verifying continuations
            verify_mode: Mode for verifying ('similarity', 'divergent', or 'skip')
            max_new_tokens: Maximum number of new tokens to generate for continuation
            temperature: Temperature for sampling
            top_p: Top-p probability threshold for nucleus sampling
            num_samples: Number of samples to generate for each divergent point
            num_continuation: Number of continuations to generate (number of times to encounter EOS tokens)
            previous_context: Number of previous sentences to include in context
            batch_size: Number of divergent points to batch together for verify
        """
        self.comparison_model = "reference"

        self.tokenizer = tokenizer
        self.verify_mode = verify_mode
        self.continuation_max_new_tokens = continuation_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.num_samples = num_samples
        self.num_continuation = num_continuation
        self.previous_context = previous_context

        self.stress_format = lambda x: "<<" + x + ">>"

        self.is_stress_mismatch = True
        self.verifier_max_new_tokens = verifier_max_new_tokens

        # InitializeMODEL_DICTn controller
        self.gen_controller = ModelController(
            comparison_model=self.comparison_model,
            mem_fraction_static=MODEL_DICT["continuation_main"]["mem_fraction_static"],
            tp_size=MODEL_DICT["continuation_main"]["tp_size"],
            base_gpu_id_main=MODEL_DICT["continuation_main"]["base_gpu_id"],
            base_gpu_id_reference=MODEL_DICT["continuation_reference"]["base_gpu_id"]
        )

        self.verify_model = VerifyModel(
            model_name=verify_model_name,
            verify_mode=verify_mode,
            max_new_tokens=self.verifier_max_new_tokens,
            mem_fraction_static=MODEL_DICT["verify"]["mem_fraction_static"],
            tp_size=MODEL_DICT["verify"]["tp_size"],
            base_gpu_id=MODEL_DICT["verify"]["base_gpu_id"]
        )

    def get_token_labels(
        self, 
        contexts: List[List[int]],
        quick_tokens: List[int],
        ref_tokens: List[int]
    ) -> Tuple[List[int], List[int], List[Union[ComparisonPoint, None]]]:
        """
        Determine if tokens diverge, generate continuations, and compare them using the verify model.
        Processes multiple token pairs in batch, sending all non-matched tokens together to generate_continuation
        and batching the diverge point comparison.
        
        Args:
            context_tokens: List of tokenized contexts
            quick_tokens: Tokens from quick model
            ref_tokens: Tokens from reference model
            
        Returns:
            Tuple of (final_tokens, token_types, comparison_points)
            where token_types is a list of TOKEN.MATCH or TOKEN.DIVERGENT,
            and comparison_points is a list of verify's comparisons (or None if tokens match)
        """
        batch_size = len(contexts)
        final_tokens = [None] * batch_size
        token_types = [None] * batch_size
        comparison_points = [None] * batch_size
        
        # First handle matching tokens and identify non-matching tokens
        non_match_indices = []
        non_match_contexts = []
        non_match_quick_tokens = []
        non_match_ref_tokens = []
        
        for i in range(batch_size):
            if quick_tokens[i] == ref_tokens[i]:
                final_tokens[i] = quick_tokens[i]
                token_types[i] = TOKEN.MATCH
                comparison_points[i] = None
            else:
                non_match_indices.append(i)
                non_match_contexts.append(contexts[i])
                non_match_quick_tokens.append(quick_tokens[i])
                non_match_ref_tokens.append(ref_tokens[i])
        
        # If no non-matching tokens, return early
        if not non_match_indices:
            return final_tokens, token_types, comparison_points
        
        # Process all non-matched tokens together
        # Generate small model continuations for all non-matched tokens
        small_outputs = self.gen_controller.generate_continuation(
            update_context_tokens=non_match_contexts,
            current_token=non_match_quick_tokens,
            model_type='small',
            past_key_values=None,
            max_new_tokens=self.continuation_max_new_tokens,
            temperature=self.temperature,
            num_samples=self.num_samples,
            top_p=self.top_p,
            num_continuation=self.num_continuation
        )
        
        # Generate reference model continuations for all non-matched tokens
        ref_outputs = self.gen_controller.generate_continuation(
            update_context_tokens=non_match_contexts,
            current_token=non_match_ref_tokens,
            model_type='reference',
            past_key_values=None,
            max_new_tokens=self.continuation_max_new_tokens,
            temperature=0.0,
            num_samples=1,
            num_continuation=self.num_continuation
        )
        
        # Collect all diverge points for batch processing
        all_diverge_points = []
        # Keep track of which diverge points belong to which token
        token_to_diverge_points = {idx: [] for idx in range(len(non_match_indices))}
        diverge_point_counter = 0
        
        # Process results for each non-matching token
        for batch_idx, orig_idx in enumerate(non_match_indices):
            context = non_match_contexts[batch_idx]
            quick_token = non_match_quick_tokens[batch_idx]
            ref_token = non_match_ref_tokens[batch_idx]
            
            # Decode tokens to text
            context_text = self.tokenizer.decode(context)
            quick_text = self.tokenizer.decode([quick_token])
            ref_text = self.tokenizer.decode([ref_token])
            
            # Use dummy values for tracking
            data_id = 0
            token_id = 0
            
            # Get reference outputs for this item
            ref_output = ref_outputs[batch_idx]
            
            # Check if we should consider this as "next context" (crossing EOS)
            is_next_context = (
                quick_token in self.gen_controller.stop_token_ids
                ) and (
                context[-1] in self.gen_controller.stop_token_ids
            )
            
            is_skip_input = False

            if self.is_stress_mismatch:
                ref_text = self.stress_format(ref_text)
                quick_text = self.stress_format(quick_text)
            
            # Get context for divergent continuations
            common_context_ref, reference_diverge_context = self.gen_controller.get_latest_context(
                context_text=context_text,
                pred_text=ref_text,
                model_output_text=ref_output["generated_text"],
                is_next_context=is_next_context,
                num_continuation=self.num_continuation,
                previous_context=self.previous_context,
                is_skip_input=is_skip_input
            )
            
            # Get small model outputs for this batch item and each sample
            small_outputs_for_item = [small_outputs[s * len(non_match_indices) + batch_idx] 
                                    for s in range(self.num_samples)]
            
            # Process each small model output for comparison; Loop though num samples
            for small_output in small_outputs_for_item:
                common_context, small_diverge_context = self.gen_controller.get_latest_context(
                    context_text=context_text,
                    pred_text=quick_text,
                    model_output_text=small_output['generated_text'],
                    is_next_context=is_next_context,
                    num_continuation=self.num_continuation,
                    previous_context=self.previous_context,
                    is_skip_input=is_skip_input
                )
                
                if common_context != common_context_ref:
                    print(f"Context mismatch: quick[...{common_context[-20:]}] vs ref[...{common_context_ref[-20:]}]")
                
                # Create diverge point for verify model
                diverge_point = DivergePoint(
                    data_id=data_id,
                    token_id=token_id,
                    pred_small_token=quick_token,
                    pred_small_text=quick_text,
                    small_diverge_text=small_diverge_context,
                    reference_diverge_text=reference_diverge_context,
                    common_context=common_context
                )
                diverge_point.pred_ref_token = ref_token
                diverge_point.pred_ref_text = ref_text
                
                # Add to overall list and track which batch item it belongs to
                all_diverge_points.append(diverge_point)
                token_to_diverge_points[batch_idx].append(diverge_point_counter)
                diverge_point_counter += 1
        
        # Batch compare all diverge points at once
        all_comparison_points = self.verify_model.batch_compare_diverge_points(all_diverge_points)
        
        # Process comparison results for each non-matching token
        for batch_idx, orig_idx in enumerate(non_match_indices):
            quick_token = non_match_quick_tokens[batch_idx]
            ref_token = non_match_ref_tokens[batch_idx]
            
            # Get comparison points for this token
            diverge_point_indices = token_to_diverge_points[batch_idx]
            comparison_points_for_token = [all_comparison_points[i] for i in diverge_point_indices]
            
            # Use the first comparison point as the representative one
            comparison_point = comparison_points_for_token[0]
            
            # Calculate average score
            avg_score = sum([point.similarity_score for point in comparison_points_for_token]) / len(comparison_points_for_token)
            
            if avg_score < 0.5:
                final_tokens[orig_idx] = quick_token
                token_types[orig_idx] = TOKEN.NEUTRAL
                comparison_points[orig_idx] = comparison_point
            else:
                final_tokens[orig_idx] = ref_token
                token_types[orig_idx] = TOKEN.DIVERGENT
                comparison_points[orig_idx] = comparison_point
        
        return final_tokens, token_types, comparison_points

    def batch_process_divergent_points(
        self,
        contexts: List[List[int]],
        quick_tokens: List[int],
        ref_tokens: List[int],
    ) -> Tuple[List[int], List[int], List[Optional[ComparisonPoint]]]:
        """
        Process multiple divergent points in batches.
        
        Args:
            contexts: List of context token sequences
            quick_tokens: List of tokens from quick model
            ref_tokens: List of tokens from reference model
            
        Returns:
            Tuple of (final_tokens, token_types, comparison_points)
        """
        # Simply call the batch-enabled get_token_label function
        return self.get_token_labels(
            contexts=contexts,
            quick_tokens=quick_tokens,
            ref_tokens=ref_tokens
        )

    def shutdown(self):
        """Clean up resources."""
        if self.gen_controller:
            self.gen_controller.shutdown()
        if self.verify_model:
            self.verify_model.shutdown()
