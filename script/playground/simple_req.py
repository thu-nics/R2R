import requests

url = f"http://localhost:30000/v1/chat/completions"

data = {
    "model": "default",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
}

response = requests.post(url, json=data)
print("Response:\n", response.json()["choices"][0]["message"]["content"])
print("LLM Ratio: ", response.json()["llm_ratio"])