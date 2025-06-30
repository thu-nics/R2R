from typing import List, Dict, Set, Optional, Union
from transformers import PreTrainedTokenizer
import re
import string
NUM_TOKEN_LENGTH = 5

def get_id_to_token_mapping(tokenizer: PreTrainedTokenizer) -> Dict[int, str]:
    
    id_to_token = {}
    vocab_size = tokenizer.vocab_size
    
    for token_id in range(vocab_size):
        token_text = tokenizer.decode([token_id])
        id_to_token[token_id] = token_text
    
    return id_to_token


def save_id_to_token_mapping(tokenizer: PreTrainedTokenizer, output_file: str) -> None:

    import json
    
    id_to_token = get_id_to_token_mapping(tokenizer)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(id_to_token, f, ensure_ascii=False, indent=2)
    
    print(f"ID to token mapping saved to {output_file}")
    print(f"Total tokens: {len(id_to_token)}")


def save_semantic_tokens_config(tokenizer: PreTrainedTokenizer, output_file: str, tokenizer_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B") -> None:
    """
    Save the semantic segmentation tokens configuration to a JSON file.
    
    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to analyze
        output_file (str): Path to output JSON file
    """
    import json
    
    # Get semantic tokens
    semantic_tokens = find_semantic_segmentation_tokens(tokenizer)
    
    # Get ID to token mapping for reference
    id_to_token = get_id_to_token_mapping(tokenizer)
    
    # Create simple mapping like id_to_token.json format
    semantic_tokens_mapping = {}
    for token_id in semantic_tokens["all_endings"]:
        semantic_tokens_mapping[token_id] = id_to_token[token_id]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"tokenizer_name": tokenizer_name, "semantic_tokens_mapping": semantic_tokens_mapping}, f, ensure_ascii=False, indent=2)
    
    print(f"Semantic tokens config saved to {output_file}")
    print(f"Total semantic tokens: {len(semantic_tokens_mapping)}")


def is_special_case_token(token_text: str) -> bool:
    """
    Check if a token should be excluded based on special cases.
    
    Special cases to exclude:
    1. Tokens containing Chinese or English colons (:, ：)
    
    Args:
        token_text (str): The token text to check
        
    Returns:
        bool: True if the token should be excluded due to special cases
    """
    # Check for colons (Chinese and English)
    colons = {':', '：'}
    if any(char in colons for char in token_text):
        return True
    
    return False


def is_valid_bracket_token(token_text: str) -> bool:
    """
    Check if a token has valid bracket structure.
    Valid cases:
    1. Balanced brackets (e.g., "()", "[]", "{}")
    2. Only right brackets (e.g., ")", "]", "}")
    3. No brackets at all
    
    Invalid cases:
    1. Only left brackets (e.g., "(", "[", "{")
    2. Special cases (colons, mismatched brackets from different sources)
    
    Args:
        token_text (str): The token text to check
        
    Returns:
        bool: True if the token has valid bracket structure
    """
    # First check for special cases
    if is_special_case_token(token_text):
        return False
    
    # Define bracket pairs using dictionaries for cleaner mapping
    bracket_pairs = {
        # English brackets
        '(': ')', '[': ']', '{': '}', '<': '>',
        # Chinese brackets
        '（': '）', '【': '】', '《': '》', '〈': '〉',
        # Quote brackets
        '"': '"', "'": "'", '「': '」', '『': '』'
    }
    
    # Create reverse mapping for right brackets
    right_brackets = set(bracket_pairs.values())
    left_brackets = set(bracket_pairs.keys())
    
    # Direct bracket pairing check
    for i, char in enumerate(token_text):
        if char in left_brackets:
            # Find the corresponding right bracket
            expected_right = bracket_pairs[char]
            # Check if the right bracket exists after the left bracket
            if expected_right not in token_text[i+1:]:
                return False
    
    # Count brackets for final validation
    left_count = sum(1 for char in token_text if char in left_brackets)
    right_count = sum(1 for char in token_text if char in right_brackets)
    
    # Valid cases:
    if left_count == 0 and right_count == 0:
        return True
    
    if left_count == 0 and right_count > 0:
        return True
    
    if left_count == right_count:
        return True
    
    # Invalid case: only left brackets
    return False

def find_semantic_segmentation_tokens(tokenizer: PreTrainedTokenizer) -> Dict[str, List[int]]:

    # Define sentence-ending symbols
    sentence_endings = ['.', '。', '!', '！', '?', '？', '\n']
    
    semantic_tokens = {
        "all_endings": []
    }
    
    vocab_size = tokenizer.vocab_size
    
    # Step 1: Find all length-1 sentence-ending symbols
    for token_id in range(vocab_size):
        try:
            token_text = tokenizer.decode([token_id])
            
            if len(token_text) == 1 and token_text in sentence_endings:
                semantic_tokens["all_endings"].append(token_id)
                
        except Exception:
            continue
    
    for token_length in range(NUM_TOKEN_LENGTH):
        for token_id in range(vocab_size):
            try:
                token_text = tokenizer.decode([token_id])
                if len(token_text) == token_length:
                    if token_text[-1] in sentence_endings and is_valid_bracket_token(token_text):
                        semantic_tokens["all_endings"].append(token_id)
                            
            except Exception:
                continue
    
    # Remove duplicates and sort all lists
    for key in semantic_tokens:
        semantic_tokens[key] = list(set(semantic_tokens[key]))
        semantic_tokens[key].sort()
    
    return semantic_tokens