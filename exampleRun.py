import requests

url = "http://localhost:8000/estimate"
data = {
    "input_tokens": 50000,
    "cached_tokens": 40000,
    "output_tokens": 800
}

response = requests.post(url, json=data)
print(response.json())
