import requests

url = "http://localhost:8000/estimate"
data = {
    "input_tokens": 10000, # Fresh input tokens
    "cached_tokens": 40000, # Cached tokens (now additive)
    "output_tokens": 800
}

response = requests.post(url, json=data)
print(response.json())
