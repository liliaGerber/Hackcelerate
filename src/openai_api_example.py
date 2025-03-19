import requests

API_KEY = "See https://starthack.eu/#/case-details?id=21, Case Description"
URL = "https://api.openai.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello, how are you?"}]
}

response = requests.post(URL, headers=headers, json=data)

print(response.json())
