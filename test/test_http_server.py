import requests
import time


# text = """<｜begin▁of▁sentence｜><｜User｜>Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.<｜Assistant｜><think>
# """
input_ids = [151646, 151646, 151644, 11510, 6556, 362, 7755, 5780, 369, 264, 400, 24, 3, 12, 85526, 20408, 23791, 4227, 323, 17933, 518, 264, 10799, 8061, 26807, 13, 3197, 1340, 22479, 518, 264, 6783, 4628, 315, 400, 82, 3, 40568, 817, 6460, 11, 279, 4227, 4990, 1059, 220, 19, 4115, 11, 2670, 400, 83, 3, 4420, 7391, 304, 279, 10799, 8061, 13, 3197, 1340, 22479, 400, 82, 10, 17, 3, 40568, 817, 6460, 11, 279, 4227, 4990, 1059, 220, 17, 4115, 323, 220, 17, 19, 4420, 11, 2670, 400, 83, 3, 4420, 7391, 304, 279, 10799, 8061, 13, 82610, 362, 7755, 22479, 518, 400, 82, 41715, 37018, 90, 16, 15170, 17, 31716, 40568, 817, 6460, 13, 7379, 279, 1372, 315, 4420, 279, 4227, 4990, 1059, 11, 2670, 279, 400, 83, 3, 4420, 7391, 304, 279, 10799, 8061, 13, 151645, 151648, 198]
url = "http://0.0.0.0:30005/generate"
data = {
    # "text": text,
    "input_ids": input_ids,
    "sampling_params": {
        "temperature": 0,
        "top_p": 1,
        "top_k": -1,
        "max_new_tokens": 2048,
    },
    "display_progress": False,
}
# Hint: You need to add chat template by yourself if using chat model here
try:
    start_time = time.time()
    response = requests.post(url, json=data)
    end_time = time.time()

    total_tokens = len(response.json().get("output_ids", ""))
    print(f"Generation completed in {end_time - start_time:.2f} seconds, total_tokens: {total_tokens}, speed: {total_tokens/(end_time - start_time):.2f}.")
    # print(response.json())
    response.raise_for_status()

except requests.exceptions.RequestException as e:
    print(f"Request error: {e}")
except Exception as e:
    print(f"Error: {e}")
