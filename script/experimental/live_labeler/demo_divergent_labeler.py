#!/usr/bin/env python3
import argparse
from transformers import AutoTokenizer
from typing import List

from r2r.data.live_labeler import LiveDivergentLabeler
from r2r.utils.config import MODEL_DICT, TOKEN

def demo_get_model_preference(text: str, tokenizer_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    """
    Demonstrates how to use the LiveDivergentLabeler's get_final_token method
    
    Args:
        text: Input text to use for demonstration
        tokenizer_name: Name of the tokenizer to use
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DICT['quick']['model_path'])
    live_labeler = LiveDivergentLabeler(
        tokenizer=tokenizer,
        verify_model_name=MODEL_DICT['verify']['model_path'],
        verify_mode="common_context",
        continuation_max_new_tokens=128,
        num_samples=1,
        num_continuation=1,
        previous_context=0
    )
    
    # Tokenize the input text
    context_tokens = tokenizer.encode(text)
    print(f"Encoded text: {text}")
    print(f"Token count: {len(context_tokens)}")
    
    # Demo 1: Matching tokens
    print("\nDEMO 1: Matching tokens")
    # Use the same token for both quick and reference models
    token_id = tokenizer.encode(" next", add_special_tokens=False)[0]
    
    print(f"Quick token: {token_id} ('{tokenizer.decode([token_id])}')")
    print(f"Reference token: {token_id} ('{tokenizer.decode([token_id])}')")
    
    final_token, token_type, comparison_point = live_labeler.get_token_labels(
        contexts=context_tokens,
        quick_token=token_id,
        ref_token=token_id
    )
    
    print(f"Result: final_token={final_token}, token_type={token_type}")
    print(f"Comparison point: {comparison_point is None}")
    
    # Demo 2: Divergent tokens
    print("\nDEMO 2: Divergent tokens")
    # Use different tokens for quick and reference models
    quick_token_id = tokenizer.encode(" quickly", add_special_tokens=False)[0]
    ref_token_id = tokenizer.encode(" slowly", add_special_tokens=False)[0]
    
    print(f"Quick token: {quick_token_id} ('{tokenizer.decode([quick_token_id])}')")
    print(f"Reference token: {ref_token_id} ('{tokenizer.decode([ref_token_id])}')")
    
    print("Calling get_final_token...")
    final_token, token_type, comparison_point = live_labeler.get_token_labels(
        contexts=context_tokens,
        quick_token=quick_token_id,
        ref_token=ref_token_id
    )
    
    print(f"Result: final_token={final_token}, token_type={token_type}")
    print(f"Token type is divergent: {token_type == TOKEN.DIVERGENT}")
    
    if comparison_point:
        print("\nComparison results:")
        print(f"Similarity score: {comparison_point.similarity_score}")
        print(f"Small diverge text excerpt: '{comparison_point.small_diverge_text[:50]}...'")
        print(f"Reference diverge text excerpt: '{comparison_point.reference_diverge_text[:50]}...'")
    
    # Clean up resources
    live_labeler.shutdown()
    print("\nDemo completed successfully")

def main():
    parser = argparse.ArgumentParser(description="Demo for LiveDivergentLabeler's get_final_token method")
    parser.add_argument("--text", type=str, default="The student walked", 
                        help="Input text to use for the demonstration")
    parser.add_argument("--tokenizer", type=str, default="mistralai/Mistral-7B-v0.1",
                        help="Name of the tokenizer to use")
    args = parser.parse_args()
    
    demo_get_model_preference(args.text, args.tokenizer)

if __name__ == "__main__":
    main() 