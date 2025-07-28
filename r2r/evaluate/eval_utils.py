from typing import Dict, List, Tuple, Callable, Any, Optional
import re
from datasets import load_dataset
import json
import base64
import zlib
import pickle
import numpy as np
## USED FOR GPQA ###

QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"


## USED FOR MMLU-PRO ###
choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]

def check_answer_correctness(predicted: str, actual: str, answer_type: str) -> bool:
    """Check if the predicted answer is correct based on the answer type."""
    if answer_type == "boxed":
        # For boxed numerical answers, compare as strings to allow for formatting differences
        return str(predicted).strip() == str(actual).strip()
    elif answer_type == "multiple_choice":
        # For multiple choice, compare uppercase letters
        return predicted.upper() == actual.upper()
    elif answer_type == "code":
        # For code answers, we'll need to run test cases - for now just check if code is not empty
        # TODO: Implement actual code testing logic
        return bool(predicted.strip())
    elif answer_type == "mmlu-multiple-choice":
        return predicted.upper() == actual.upper()
    elif answer_type == "livecodebench":
        return True
    else:
        raise ValueError(f"Unsupported answer type for correctness check: {answer_type}")

def get_answer_extractor(dataset_type: str) -> Callable:
    """Return the appropriate answer extraction function based on dataset type."""
    extractors = {
        "boxed": extract_boxed_answer,
        "multiple_choice": extract_multiple_choice_answer,
        "livecodebench": dummy_extract_code_answer,
        "mmlu-multiple-choice": extract_mmlu_pro_answer
    }
    
    if dataset_type in extractors:
        return extractors[dataset_type]
    else:
        raise ValueError(f"Unsupported dataset answer type: {dataset_type}")
    
def extract_boxed_answer(text: str) -> Tuple[str, bool]:
    """Extract answer from \boxed{...} and check if it's a valid number."""
    pattern = r"\\boxed{([^}]*)}"
    match = re.search(pattern, text)
    if match:
        answer = match.group(1).strip()
        # Try to convert to int, return None if fails
        try:
            return answer, True
        except:
            return answer, False
    return "", False

def extract_multiple_choice_answer(text: str) -> Tuple[str, bool]:
    """Extract multiple-choice answer (A, B, C, or D) from the generated text."""
    # Look for answer statements like "The answer is A" or "I choose B"
    
    match = re.search(ANSWER_PATTERN_MULTICHOICE, text)
    if match:
        choice = match.group(1).upper()  # Convert to uppercase
        return choice, True
    
    return "", False

def dummy_extract_code_answer(text: str) -> Tuple[str, bool]:
    """Dummy function to extract code from the generated text."""
    return text, False

def extract_code_answer(text: str) -> Tuple[str, bool]:
    """Extract Python code from the generated text."""
    # Look for code blocks marked with ```python or ``` markers
    code_block_patterns = [
        r"```python\n(.*?)```",  # Python-specific code blocks
        r"```\n(.*?)```",        # Generic code blocks
        r"`{3,}(.*?)`{3,}"      # Any code blocks with 3 or more backticks
    ]
    
    for pattern in code_block_patterns:
        matches = re.finditer(pattern, text, re.DOTALL)
        for match in matches:
            code = match.group(1).strip()
            if code:
                return code, True
    
    # If no code blocks found, try to find Python-like code directly
    lines = text.split('\n')
    code_lines = []
    in_code = False
    
    for line in lines:
        # Skip empty lines at the start
        if not in_code and not line.strip():
            continue
            
        # Look for common Python code indicators
        if not in_code and (
            line.strip().startswith('def ') or
            line.strip().startswith('class ') or
            line.strip().startswith('import ') or
            line.strip().startswith('from ') or
            ':' in line
        ):
            in_code = True
            
        if in_code:
            code_lines.append(line)
            
    if code_lines:
        return '\n'.join(code_lines), True
        
    return "", False

def extract_mmlu_pro_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1), True
    else:
        return extract_mmlu_pro_answer_again(text)


def extract_mmlu_pro_answer_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1), True
    else:
        return extract_mmlu_pro_answer_final(text)


def extract_mmlu_pro_answer_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0), True
    else:
        print("answer extract failed\n")
        return None, False

def preprocess(test_df):
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    return res_df

def select_by_category(df, subject):
    res = []
    for each in df:
        if each["category"] == subject:
            res.append(each)
    return res

def format_cot_example(example, including_answer=True):
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt)
    if including_answer:
        cot_content = example["cot_content"].replace("A: Let's think step by step.",
                                                     "Answer: Let's think step by step.")
        prompt += cot_content + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


def generate_cot_prompt(val_df, curr, k):
    prompt = ""
    with open(f"script/evaluate/eval_configs/mmlu-pro_initial_prompt.txt", "r") as fi:
        for line in fi.readlines():
            prompt += line
    subject = curr["category"]
    val_df = select_by_category(val_df, subject)
    val_df = val_df[: k]
    prompt = prompt.replace("{$}", subject) + "\n"
    for example in val_df:
        prompt += format_cot_example(example, including_answer=True)
    prompt += format_cot_example(curr, including_answer=False)
    return prompt

def prepare_prompt(line: dict[str, Any]) -> str:
    query = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
    query += f"Question: {line['question_content']}\n\n"
    if starter_code := line.get("starter_code", None):
        query += "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
        query += f"```python\n{starter_code}\n```\n\n"
    else:
        query += "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows."
        query += "```python\n# YOUR CODE HERE\n```\n\n"
    return query


def lcb_codegeneration_prompt_fn(line):
    # For the prompt we need a more general function that can be used tweaked like in:
    # https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/prompts/code_generation.py
    query = prepare_prompt(line)
    # List of dicts of the form: [{"input": "6\nabc\nacb\nbac\nbca\ncab\ncba\n", "output": "YES\nYES\nYES\nNO\nNO\nYES\n", "testtype": "stdin"}]
    public_test_cases = json.loads(line["public_test_cases"])
    private_test_cases = translate_private_test_cases(line["private_test_cases"])
    inputs = [test["input"] for test in public_test_cases + private_test_cases]
    outputs = [test["output"] for test in public_test_cases + private_test_cases]
    
    return query, inputs, outputs

def translate_private_test_cases(encoded_data: str) -> dict[str, str]:
    decoded_data = base64.b64decode(encoded_data)
    decompressed_data = zlib.decompress(decoded_data)
    original_data = pickle.loads(decompressed_data)
    return json.loads(original_data)

def prepare_multiple_choice_prompt(line: dict[str, Any], format_config: dict[str, Any]) -> tuple[str, str]:
    
    options_fields = format_config.get("options_fields", [])
    if len(options_fields) >= 4:  # Need at least 4 options for A, B, C, D
        # Get the options in a consistent order
        options = [line[field] for field in options_fields]
                
        # Shuffle the options to randomize the correct answer position
        # Create a mapping from original positions to shuffled positions
        indices = list(range(len(options)))
        np.random.shuffle(indices)
                
        shuffled_options = [options[i] for i in indices]
                
        # Find where the correct answer ended up
        correct_index = indices.index(0)  # Assuming the first option is the correct one
        correct_letter = chr(65 + correct_index)  # A, B, C, D...
                
        options = {
            "A": shuffled_options[0],
            "B": shuffled_options[1],
            "C": shuffled_options[2],
            "D": shuffled_options[3]
        }
        answer = correct_letter

        # Format the problem with options
        formatted_problem = QUERY_TEMPLATE_MULTICHOICE.format(
            Question=line[format_config["question_field"]],
            A=options["A"],
            B=options["B"],
            C=options["C"],
            D=options["D"]
        )

        return formatted_problem, answer
