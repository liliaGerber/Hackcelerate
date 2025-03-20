from ollama import chat

stream = chat(
    model="gemma3:1b",
    messages=[
        {
            "role": "user",
            "content": "Answer the question asked by user"
            "Why is the sky blue? "
            "Give reply based on personality traits without mentioning about it in response"
            "EXT:77.18 NEU: 61.74 AGR: 75.51 CON:70.34 OPN:80.39",
        }
    ],
    stream=True,
)

for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)
