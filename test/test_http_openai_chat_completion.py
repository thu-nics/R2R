import requests
import time

url = "http://0.0.0.0:30000/v1/chat/completions"

# OpenAI-compatible chat completion request
data = {
    "model": "default",
    "messages": [
        {
            "role": "user",
            "content": "Every morning Aya goes for a 9-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of s kilometers per hour, the walk takes her 4 hours, including t minutes spent in the coffee shop. When she walks s+2 kilometers per hour, the walk takes her 2 hours and 24 minutes, including t minutes spent in the coffee shop. Suppose Aya walks at s+1/2 kilometers per hour. Find the number of minutes the walk takes her, including the t minutes spent in the coffee shop."
        }
    ],
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 2048,
    "stream": False
}

try:
    start_time = time.time()
    response = requests.post(url, json=data)
    end_time = time.time()

    response.raise_for_status()
    
    try:
        result = response.json()
        
        # Extract token counts from usage info
        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        
        # Extract generated text from choices
        choices = result.get("choices", [])
        if choices:
            generated_text = choices[0].get("message", {}).get("content", "")
            finish_reason = choices[0].get("finish_reason", "")
        else:
            generated_text = ""
            finish_reason = ""
        
        # Uncomment to see the full response
        print(f"\nGenerated text:\n{generated_text}")

        print(f"Chat completion completed in {end_time - start_time:.2f} seconds")
        print(f"Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}, Total tokens: {total_tokens}")
        print(f"Speed: {completion_tokens/(end_time - start_time):.2f} tokens/s")
        print(f"Finish reason: {finish_reason}")
        print(f"Model: {result.get('model', 'unknown')}")
        print(f"Response ID: {result.get('id', 'unknown')}")
        
        
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
