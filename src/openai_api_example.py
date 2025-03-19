import requests

API_KEY = "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"
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
