import requests
import time

url = "http://0.0.0.0:30000/health"

# Hint: You need to add chat template by yourself if using chat model here
try:
    start_time = time.time()
    response = requests.get(url)
    end_time = time.time()

    response.raise_for_status()
    
    try:
        print(f"Response text: {response.text}")
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
