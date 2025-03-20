from ollama import chat
import sounddevice as sd
import pyttsx3
import os

stream = chat(
    model="gemma3:1b",
    messages=[
        {
            "role": "user",
            "content": "Answer the question asked by user"
            "Can you give me some food recommendations for today"
            "Give reply based on personality traits without mentioning about it in response"
            "EXT:1 NEU: 1 AGR: 1 CON: 0 OPN: 0",
        }
    ],
    stream=True,
)


response_content=""

for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)
    chunk_text = chunk["message"]["content"]
    response_content += chunk_text

#Text to speech configuration
engine = pyttsx3.init()  # Object creation
# Setting a new speaking rate
engine.setProperty('rate', 200)

# Selecting a voice (0 for male, 1 for female, etc.)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

text = str(response_content)
print(text)

engine.say(response_content)
# Run the speech engine
engine.runAndWait()

# Save to a file
#engine.save_to_file(response_content, 'response.mp3')
#engine.runAndWait()

# Stop the engine
engine.stop()
