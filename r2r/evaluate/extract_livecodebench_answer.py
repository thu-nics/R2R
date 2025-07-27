import pandas as pd
import argparse
from r2r.evaluate.codegen_metrics import codegen_metrics, extract_code
import json
import ast
from typing import List, Optional, Tuple

class Doc:
    
    specific: dict = None  # Information which is specific to the current eval
    
    def __init__(self, specific: dict):
    
        self.specific = specific


def codegen_metric(predictions: List[str], formatted_doc: Doc, **kwargs) -> Tuple[float, bool]:
    """Estimates the Pass@1 metric for the code generation task.
    Extract the code from each prediction, Runs it for each sample and generations,
    and computes the Pass@1 over the outputs.
    """
        
    generated_code_snippets = [[extract_code(pred) for pred in predictions]]  # noqa: F841
    
    if len(generated_code_snippets[0][0]) > 0:
        has_extracted_answer = True
    else:
        has_extracted_answer = False
    
    evaluation_sample = {  # noqa: F841
        "inputs": formatted_doc.specific["inputs"],
        "outputs": formatted_doc.specific["outputs"],
        "fn_name": formatted_doc.specific["fn_name"],
    }
    # This is a list of lists because
    evaluation_sample = [{"input_output": json.dumps(evaluation_sample)}]

    metrics, _ = codegen_metrics(
        evaluation_sample,
        generated_code_snippets,
        k_list=[1],  # Only run for Pass@1
        num_process_evaluate=1,
    )
    return metrics["pass@1"], has_extracted_answer

def read_csv_to_df(csv_path='output/livecodebench/1.5B/combined_results.csv') -> Optional[pd.DataFrame]:
    """
    Read a CSV file into a pandas DataFrame.
    
    Args:
        csv_path (str): Path to the CSV file. Default is 'output/livecodebench/1.5B/combined_results.csv'
    
    Returns:
        Optional[pd.DataFrame]: The loaded DataFrame or None if there's an error
    """
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print("Warning: The CSV file is empty")
            return None
        required_columns = ['correct_answer', 'predicted_answer']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return None
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return None

def convert_string_to_dict(string_dict: str) -> Optional[dict]:
    """
    Convert a string representation of a dictionary to a Python dictionary.
    
    Args:
        string_dict (str): String representation of a dictionary
    
    Returns:
        Optional[dict]: The converted dictionary or None if conversion fails
    """
    if not isinstance(string_dict, str):
        print(f"Error: Expected string, got {type(string_dict)}")
        return None
        
    try:
        # First try using ast.literal_eval as it's safer than eval
        return ast.literal_eval(string_dict)
    except (SyntaxError, ValueError):
        try:
            # If ast.literal_eval fails, try json.loads
            return json.loads(string_dict)
        except json.JSONDecodeError:
            print(f"Error: Could not convert string to dictionary: {string_dict[:100]}...")
            return None

def calculate_metrics(item: pd.Series) -> Tuple[Optional[float], bool]:
    """
    Process a single row from the DataFrame.
    
    Args:
        item (pd.Series): A row from the DataFrame
    
    Returns:
        Optional[float]: The codegen metric result or None if processing fails
    """
    try:
        specific_dict = convert_string_to_dict(item['correct_answer'])
        if specific_dict is None:
            return None
            
        extracted_Doc = Doc(specific_dict)
        # Wrap the predicted answer in a list since codegen_metric expects a list of predictions
        codegen_metric_result, has_extracted_answer = codegen_metric([item["predicted_answer"]], extracted_Doc)
        return codegen_metric_result, has_extracted_answer
    except Exception as e:
        print(f"Error processing row: {str(e)}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read CSV file into DataFrame')
    parser.add_argument('--csv_path', type=str, 
                      default='output/final_eval/golden/0508/livecodebench/default/new_combined_results.csv',
                      help='Path to the CSV file')
    
    args = parser.parse_args()
    origin_path = args.csv_path.split('/')[0:-1]
    origin_path = '/'.join(origin_path)
    df = read_csv_to_df(args.csv_path)
    
    if df is not None:
        is_correct_list = []
        has_extracted_answer_list = []
        for idx, item in df.iterrows():
            result, has_extracted_answer = calculate_metrics(item)
            has_extracted_answer_list.append(has_extracted_answer)
            if result is not None:
                is_correct_list.append(result)
                print(f"Row {idx}: Codegen metric result: {result}")
        
        df['is_correct'] = is_correct_list
        df['has_extracted_answer'] = has_extracted_answer_list
        
        if is_correct_list:
            avg_result = sum(is_correct_list) / len(is_correct_list)
            print(f"\nAverage codegen metric result: {avg_result}")
        else:
            print("No valid results were obtained from any rows")
    
    df = df.drop(columns=['correct_answer'])

    df.to_csv(f'{origin_path}/combined_results_evaluation_light.csv', index=False)
    
    
