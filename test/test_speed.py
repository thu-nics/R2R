import requests
import time
import argparse
import os
import csv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="test/input_text.txt")
    parser.add_argument("--input_id", type=int, default=0)
    parser.add_argument("--is_background", action="store_true", default=False)
    parser.add_argument("--output_csv", type=str, default="output/latency_results.csv")
    return parser.parse_args()

def append_to_csv(csv_path, input_id, speed, llm_ratio):
    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)

        # 如果文件不存在，先写表头
        if not file_exists:
            writer.writerow(["id", "speed_tokens_per_sec", "llm_ratio"])

        writer.writerow([input_id, f"{speed:.6f}", f"{llm_ratio:.6f}"])

args = parse_args()

with open(args.input_file, "r") as f:
    lines = f.readlines()
    input_id = -(args.input_id % len(lines)) if args.is_background else args.input_id % len(lines)
    input_text = lines[input_id].strip()

url = "http://0.0.0.0:30000/v1/chat/completions"

# OpenAI-compatible chat completion request
data = {
    "model": "default",
    "messages": [
        {
            "role": "user",
            "content": input_text,
        }
    ],
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 2048,
    "stream": False
}

try:
    if args.is_background:
        while True:
            response = requests.post(url, json=data)
            response.raise_for_status()
    else:
        start_time = time.time()
        response = requests.post(url, json=data)
        end_time = time.time()

        response.raise_for_status()
        
        try:
            result = response.json()
            
            # Extract token counts from usage info
            usage = result.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)
            llm_ratio = result.get("llm_ratio", 0)
            
            # Uncomment to see the full response
            speed = completion_tokens/(end_time - start_time)
            print(f"{speed:.2f} {llm_ratio:.4f}")
            append_to_csv(args.output_csv, args.input_id, speed, llm_ratio)
        
        except ValueError as e:
            print(f"Failed to parse JSON response: {e}")
            print(f"Response status code: {response.status_code}")
            print(f"Response text: {response.text[:500]}")

except requests.exceptions.RequestException as e:
    print(f"Request error: {e}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"Response status code: {e.response.status_code}")
        try:
            print(f"Response text: {e.response.text[:500]}")
        except:
            print("Could not read response text")
except ValueError as e:
    print(f"JSON parsing error: {e}")
    if 'response' in locals():
        print(f"Response status code: {response.status_code}")
        print(f"Response text: {response.text[:500]}")
except Exception as e:
    print(f"Error: {e}")
    if 'response' in locals():
        print(f"Response status code: {response.status_code}")
        print(f"Response text: {response.text[:500]}")
