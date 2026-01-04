"""
Test script for OpenAI client compatibility with the R2R HTTP server.
Uses the official OpenAI Python client library.

Make sure to install the openai package:
    pip install openai
"""

import time
from openai import OpenAI

# Configure the OpenAI client to use our local server
client = OpenAI(
    base_url="http://0.0.0.0:30000/v1",
    api_key="not-needed"  # API key is not required for local server
)

# Test messages
messages = [
    {
        "role": "user",
        "content": "Every morning Aya goes for a 9-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of s kilometers per hour, the walk takes her 4 hours, including t minutes spent in the coffee shop. When she walks s+2 kilometers per hour, the walk takes her 2 hours and 24 minutes, including t minutes spent in the coffee shop. Suppose Aya walks at s+1/2 kilometers per hour. Find the number of minutes the walk takes her, including the t minutes spent in the coffee shop."
    },
    {
        "role": "user",
        "content": "Every morning Aya goes for a 9-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of s kilometers per hour, the walk takes her 4 hours, including t minutes spent in the coffee shop. When she walks s+2 kilometers per hour, the walk takes her 2 hours and 24 minutes, including t minutes spent in the coffee shop. Suppose Aya walks at s+1/2 kilometers per hour. Find the number of minutes the walk takes her, including the t minutes spent in the coffee shop."
    },
    {
        "role": "user",
        "content": "Every morning Aya goes for a 9-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of s kilometers per hour, the walk takes her 4 hours, including t minutes spent in the coffee shop. When she walks s+2 kilometers per hour, the walk takes her 2 hours and 24 minutes, including t minutes spent in the coffee shop. Suppose Aya walks at s+1/2 kilometers per hour. Find the number of minutes the walk takes her, including the t minutes spent in the coffee shop."
    }
]

try:
    print("Testing OpenAI client compatibility...")
    print("-" * 50)
    
    start_time = time.time()
    
    # Make a chat completion request using the OpenAI client
    response = client.chat.completions.create(
        model="default",
        messages=messages,
        temperature=0,
        top_p=1,
        max_tokens=2048
    )
    
    end_time = time.time()
    
    # Extract response information
    choice = response.choices[0]
    generated_text = choice.message.content
    finish_reason = choice.finish_reason
    
    # Usage statistics
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens
    
    print(f"Chat completion completed in {end_time - start_time:.2f} seconds")
    print(f"Response ID: {response.id}")
    print(f"Model: {response.model}")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Completion tokens: {completion_tokens}")
    print(f"Total tokens: {total_tokens}")
    print(f"Speed: {completion_tokens/(end_time - start_time):.2f} tokens/s")
    print(f"Finish reason: {finish_reason}")
    print("-" * 50)
    # Uncomment to see the full response
    # print(f"\nGenerated text:\n{generated_text}")
    
    print("\n✓ OpenAI client test passed successfully!")

except Exception as e:
    print(f"✗ Error during OpenAI client test: {e}")
    import traceback
    traceback.print_exc()
