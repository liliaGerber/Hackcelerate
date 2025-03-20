from ollama import chat

stream = chat(
    model="gemma3:1b",
    messages=[
        {
            "role": "user",
            "content": "Answer the question asked by user"
            "Can you get my my favourite pizza"
            "Give reply based on personality traits without mentioning about it in response"
            "EXT:1 NEU: 1 AGR: 1 CON: 0 OPN: 0",
        }
    ],
    stream=True,
)

for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)
