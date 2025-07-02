import re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import sglang as sgl
from sglang.srt.hf_transformers_utils import get_tokenizer
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict, Any
import pandas as pd
import os

from r2r.data.data_process import MismatchPoint
from r2r.data.generation_controller import DivergePoint, ModelController
from r2r.utils.config import MODEL_DICT

DIVERGENT_SIMPLE_SYSTEM_PROMPT = "You are a semantic comparison expert."

@dataclass
class ComparisonPoint:
    """Represents a comparison point with judgment results"""
    # Identifiers
    data_id: int
    token_id: int
    
    # Prediction data
    pred_small_token: List[int]
    pred_small_text: str
    
    # Judgment data
    small_diverge_text: str
    reference_diverge_text: str
    common_context: str
    similarity_score: int = None  # Similarity score between 1-10 or 0-1 binary
    verify_response: str = None  # Response from the model

    def print(self):
        """Print comparison information in a formatted way"""
        print(f"Common context: {self.common_context}")
        print(f"Refer diverge text: {self.reference_diverge_text}")
        print(f"Small diverge text: {self.small_diverge_text}")
        print(f"Verify response: {self.verify_response}")
        print(f"Score: {self.similarity_score}")

class VerifyModel:
    """Model for judging the similarity between two text continuations"""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        verify_mode: str = "divergent",
        max_new_tokens: int = 128,
        mem_fraction_static: float = 0.5,
        tp_size: int = 2,
        base_gpu_id: int = 0,
        apply_chat_template_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.device = device
        self.model_name = model_name
        self.verify_mode = verify_mode
        self.apply_chat_template_kwargs = apply_chat_template_kwargs or {}

        print(f"Loading verify model {self.model_name}...")
        # Using HuggingFace tokenizer directly for token-based processing
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        # Initialize Engine with skip_tokenizer_init=True for token-based processing
        self.model = sgl.Engine(
            model_path=self.model_name, 
            dtype="bfloat16", 
            mem_fraction_static=mem_fraction_static,
            skip_tokenizer_init=True,
            tp_size=tp_size,
            base_gpu_id=base_gpu_id
        )

        self.max_new_tokens = max_new_tokens

        # Set system prompt based on verify_mode
        if verify_mode == "common_context":
            self.system_prompt = DIVERGENT_SIMPLE_SYSTEM_PROMPT
        else:
            raise ValueError(f"Invalid verify mode: {verify_mode}")

        print(f"Using {verify_mode} mode for verifying")

        # Create the system prompt message
        self.system_message = [{"role": "system", "content": self.system_prompt}]

    @staticmethod
    def get_divergent_user_message_common_context(text1: str, text2: str, text3: str) -> List[Dict[str, str]]:
        """Create chat messages for the comparison task"""
        divergent_user_prompt = f"""**Task:**  
Determine if the divergence between Sentence 1 and Sentence 2 affects the meaning, reasoning, logic, or conclusions derived from them.

**Instructions:**
- The marker `<< >>` indicates where the sentences diverge. It is **not** part of the original text.
- Assess whether this divergence changes the meaning, reasoning, logic, or conclusions, or if it introduces new information or contradictions.

**Output `1` if:**  
- The divergence causes a change in meaning, reasoning, logic, or conclusions.  
- It introduces new information, shifts focus, or contradicts prior facts.  
- The sentences follow different reasoning paths or focus on different aspects.

**Output `0` if:**  
- The divergence is superficial and does not affect meaning, reasoning, logic, or conclusions.  
- Both sentences follow the same reasoning path or lead to the same conclusion.

**Reasoning:** Provide a brief explanation of how the divergence impacts (or does not impact) meaning, reasoning, logic, or conclusions.

---

### Example 1 (Same - 0):  
Sentence 1: `"The ratio of adults to total people <<is>> now 11/25."`  
Sentence 2: `"The ratio of adults to total people <<chang>>ed from 5/12 to 11/25 after adding 50 people."`  
Output: 0  
Reasoning: The change from "is" to "changed" does not affect the overall meaning, reasoning, logic, or conclusions.

### Example 2 (Different - 1):  
Sentence 1: `"Let's solve this using <<integration>> by parts."`  
Sentence 2: `"Let's solve this using <<u->>substitution."`  
Output: 1  
Reasoning: The change in method (from integration by parts to substitution) alters the reasoning and approach to solving the problem.

---

### Now complete the task:

Common Context:  
\"\"\"  
{text1}  
\"\"\"  

Sentence 1:  
\"\"\"  
{text2}  
\"\"\"  

Sentence 2:  
\"\"\"  
{text3}  
\"\"\"  

**Answer (Output: <0 or 1>)**
**Reasoning:**"""
        return divergent_user_prompt

    def verify(self, text1: str, text2: str) -> Tuple[int, str]:
        """verify the similarity between two text continuations using token-based processing
        
        Args:
            text1: First text continuation
            text2: Second text continuation
            
        Returns:
            Tuple of (similarity_score, response)
            - similarity_score: Integer between 1 and 10 for similarity mode, 0 or 1 for divergent mode
            - response: Response from the model
        """
        # Get the appropriate user prompt based on verify_mode
        user_prompt = self.get_divergent_user_message(text1, text2)

        # Prepare full prompt with system and user message
        full_prompt = self.system_text + "\n\n" + user_prompt

        # Tokenize the full prompt for token-based generation
        input_token_ids = self.tokenizer.encode(full_prompt)

        # Use sgl.Engine for token-based generation
        sampling_params = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": 0.0
        }

        try:
            # Generate the response using token IDs directly
            outputs = self.model.generate(input_ids=[input_token_ids], sampling_params=sampling_params)

            # Get generated token IDs and decode to text
            output_token_ids = outputs[0]['output_ids']
            response = self.tokenizer.decode(output_token_ids)

            # Extract score from response
            match = re.search(r'\d+', response)
            score = int(match.group()) if match else -1

            # For divergent mode, only accept 0 or 1
            if self.verify_mode == "divergent" and score not in [0, 1]:
                print(f"Warning: Unexpected score {score} in divergent mode. Setting to -1.")
                score = -1

            return score, response
        except Exception as e:
            print(f"Error in token-based generation: {e}")
            return -1, str(e)

    def verify_common_context(self, text1: str, text2: str, text3: str) -> Tuple[int, str]:
        """verify the similarity between two text continuations using token-based processing
        
        Args:
            text1: Common context
            text2: Sentence 1
            text3: Sentence 2
            
        Returns:
            Tuple of (similarity_score, response)
            - similarity_score: Integer between 1 and 10 for similarity mode, 0 or 1 for divergent mode
            - response: Response from the model
        """
        # Get the appropriate user prompt based on verify_mode
        user_prompt = self.get_divergent_user_message_common_context(text1, text2, text3)

        # Prepare full prompt with system and user message
        full_prompt = self.system_text + "\n\n" + user_prompt

        # Tokenize the full prompt for token-based generation
        input_token_ids = self.tokenizer.encode(full_prompt)

        # Use sgl.Engine for token-based generation
        sampling_params = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": 0.0
        }

        try:
            # Generate the response using token IDs directly
            outputs = self.model.generate(input_ids=[input_token_ids], sampling_params=sampling_params)

            # Get generated token IDs and decode to text
            output_token_ids = outputs[0]['output_ids']
            response = self.tokenizer.decode(output_token_ids)

            # Extract score from response
            match = re.search(r'\d+', response)
            score = int(match.group()) if match else -1

            # For divergent mode, only accept 0 or 1
            if self.verify_mode == "divergent" and score not in [0, 1]:
                print(f"Warning: Unexpected score {score} in divergent mode. Setting to -1.")
                score = -1

            return score, response
        except Exception as e:
            print(f"Error in token-based generation: {e}")
            return -1, str(e)

    def compare_diverge_point(self, diverge_point: DivergePoint) -> ComparisonPoint:
        """Compare the two continuations in a DivergePoint and return a ComparisonPoint
        
        Args:
            diverge_point: DivergePoint containing the two continuations to compare
            
        Returns:
            ComparisonPoint with verify results
        """
        score, verify_response = self.verify(
            diverge_point.small_diverge_text,
            diverge_point.reference_diverge_text
        )

        return ComparisonPoint(
            data_id=diverge_point.data_id,
            token_id=diverge_point.token_id,
            pred_small_token=diverge_point.pred_small_token,
            pred_small_text=diverge_point.pred_small_text,
            small_diverge_text=diverge_point.small_diverge_text,
            reference_diverge_text=diverge_point.reference_diverge_text,
            common_context=diverge_point.common_context,
            similarity_score=score,
            verify_response=verify_response
        )

    def batch_compare_diverge_points(self, diverge_points: List[DivergePoint]) -> List[ComparisonPoint]:
        """Compare multiple diverge points in a batch using token-based processing
        
        Args:
            diverge_points: List of DivergePoints to compare
            
        Returns:
            List of ComparisonPoints with verify results
        """
        comparison_points = []

        # Prepare all prompts and tokenize them for batch processing
        input_ids_list = []
        for diverge_point in diverge_points:
            if self.verify_mode == "common_context":
                user_prompt = self.get_divergent_user_message_common_context(
                    diverge_point.common_context,
                    diverge_point.small_diverge_text,
                    diverge_point.reference_diverge_text
                )

            else:
                raise ValueError(f"Invalid verify mode: {self.verify_mode}")

            # Prepare full prompt and tokenize
            user_message = [{"role": "user", "content": user_prompt}]
            messages = self.system_message + user_message

            # Merge default kwargs with user-provided kwargs
            chat_template_kwargs = {"tokenize": False, "add_generation_prompt": True}
            chat_template_kwargs.update(self.apply_chat_template_kwargs)

            full_prompt = self.tokenizer.apply_chat_template(messages, **chat_template_kwargs)
            input_token_ids = self.tokenizer.encode(full_prompt)
            input_ids_list.append(input_token_ids)

        # Prepare sampling parameters
        sampling_params = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": 0.0
        }

        # Execute batch generation using token IDs directly
        try:
            outputs = self.model.generate(input_ids=input_ids_list, sampling_params=sampling_params)

            # Process results
            for i, (output, diverge_point) in enumerate(zip(outputs, diverge_points)):
                # Get generated token IDs and decode to text
                output_token_ids = output['output_ids']
                response = self.tokenizer.decode(output_token_ids, skip_special_tokens=True)

                # Extract score from response
                score = self._response_to_score(response, self.verify_mode)

                # Create comparison point
                comparison_point = ComparisonPoint(
                    data_id=diverge_point.data_id,
                    token_id=diverge_point.token_id,
                    pred_small_token=diverge_point.pred_small_token,
                    pred_small_text=diverge_point.pred_small_text,
                    small_diverge_text=diverge_point.small_diverge_text,
                    reference_diverge_text=diverge_point.reference_diverge_text,
                    similarity_score=score,
                    verify_response=response,
                    common_context=diverge_point.common_context
                )
                if hasattr(diverge_point, "pred_ref_token") and hasattr(diverge_point, "pred_ref_text"):
                    comparison_point.pred_ref_token = diverge_point.pred_ref_token
                    comparison_point.pred_ref_text = diverge_point.pred_ref_text

                comparison_points.append(comparison_point)

        except Exception as e:
            print(f"Error in batch token-based processing: {e}")
            # Create fallback comparison points with error for all diverge points
            for diverge_point in diverge_points:
                comparison_points.append(ComparisonPoint(
                    data_id=diverge_point.data_id,
                    token_id=diverge_point.token_id,
                    pred_small_token=diverge_point.pred_small_token,
                    pred_small_text=diverge_point.pred_small_text,
                    small_diverge_text=diverge_point.small_diverge_text,
                    reference_diverge_text=diverge_point.reference_diverge_text,
                    similarity_score=-1,
                    verify_response=f"Error: {str(e)}",
                    common_context=diverge_point.common_context
                ))

        return comparison_points

    def _response_to_score(self, response: str, verify_mode: str) -> int:
        """Convert the response to a score"""
        # Try the more specific pattern first (looking for "output" followed by digits)
        match = re.search(r'(?i)\boutput\b[^0-9]*?(\d+)', response)
        if match:
            score = int(match.group(1))
        else:
            # Fall back to simple digit extraction
            match = re.search(r'\d+', response)
            score = int(match.group()) if match else -1

        # For divergent mode, only accept 0 or 1
        if self.verify_mode == "divergent" and score not in [0, 1]:
            print(f"Warning: Unexpected score {score} in divergent mode. Setting to -1.")
            score = -1

        return score

    def shutdown(self):
        """Shut down the Engine instance to free resources"""
        try:
            self.model.shutdown()
            print(f"VerifyModel engine shut down successfully")
        except Exception as e:
            print(f"Error shutting down engine: {str(e)}")


def data_points_to_df(
    comparison_points: List[ComparisonPoint],
    mismatch_points: Dict[Tuple[int, int], MismatchPoint],
    comparison_model: str,
    is_verify: bool = True,
) -> pd.DataFrame:
    """Convert a list of ComparisonPoints to a pandas DataFrame, combining with MismatchPoint data
    
    Args:
        comparison_points: List of ComparisonPoints to convert
        mismatch_points: Dictionary of MismatchPoints by (data_id, token_id)
        comparison_model: Model to use for comparison ('reference' or 'real')
        is_verify: Whether verify model is being used. If False, don't include similarity score andverify response
        
    Returns:
        DataFrame containing comparison and mismatch data
    """
    # Convert each ComparisonPoint to a dict
    data = []
    for comparison_point in tqdm(comparison_points, desc="Converting comparison points to DataFrame", leave=False):
        # Get the corresponding mismatch point using data_id and token_id as keys
        mismatch_point = mismatch_points.get((comparison_point.data_id, comparison_point.token_id))

        if mismatch_point is None:
            # Skip if no matching mismatch point (this shouldn't happen with proper implementation)
            continue

        point_dict = {
            # Basic identifiers
            "data_id": comparison_point.data_id,
            "token_id": comparison_point.token_id,
            # Original tokens and predictions from MismatchPoint
            "real_token": mismatch_point.real_token,
            "real_text": mismatch_point.real_text,
            "pred_small_token": comparison_point.pred_small_token,
            "pred_small_text": comparison_point.pred_small_text,
            # Divergent continuations from ComparisonPoint
            "small_diverge_text": comparison_point.small_diverge_text,
            "reference_diverge_text": comparison_point.reference_diverge_text,
            "common_context": comparison_point.common_context,
        }
            
        # Only add verify results if is_verify is True
        if is_verify:
            point_dict["similarity_score"] = comparison_point.similarity_score
            point_dict["verify_response"] = comparison_point.verify_response
            
        data.append(point_dict)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Define basic columns that are always included

    column_order = [
        # Identifiers
        "data_id",
        "token_id",
        # Original and predictions
        "real_token",
        "real_text",
        "pred_small_token",
        "pred_small_text",
         # Divergent continuations
        "small_diverge_text",
        "reference_diverge_text",
        "common_context",
    ]
    
    # Add verify columns if is_verify is True
    if is_verify:
        column_order.extend(["similarity_score", "verify_response"])

    # Only include columns that exist in the DataFrame
    available_columns = [col for col in column_order if col in df.columns]
    return df[available_columns]
