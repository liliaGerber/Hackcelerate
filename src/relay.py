import io
import json
import pickle
import re
import sqlite3
import uuid

import numpy as np
import pandas as pd
import torch
from faster_whisper import WhisperModel
from flasgger import Swagger
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sock import Sock
from sentence_transformers import SentenceTransformer

from ollama import chat

app = Flask(__name__)
sock = Sock(app)
cors = CORS(app)
swagger = Swagger(app)

sessions = {}

cEXT = pickle.load(open("data/models/cEXT.p", "rb"))
cNEU = pickle.load(open("data/models/cNEU.p", "rb"))
cAGR = pickle.load(open("data/models/cAGR.p", "rb"))
cCON = pickle.load(open("data/models/cCON.p", "rb"))
cOPN = pickle.load(open("data/models/cOPN.p", "rb"))
vectorizer_31 = pickle.load(open("data/models/vectorizer_31.p", "rb"))
vectorizer_30 = pickle.load(open("data/models/vectorizer_30.p", "rb"))
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
DB_PATH = "memories.sqlite"


def init_db():
    """Initialize the SQLite database and create the memories table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_session_id TEXT,
            text TEXT,
            embedding TEXT
        )
    """)
    conn.commit()
    conn.close()


def get_embedding(text):
    # The model returns a numpy array; convert it to list so it can be stored as JSON.
    embedding = embedding_model.encode(text)
    return embedding.tolist()


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def transcribe_whisper(audio_recording):
    audio_file = io.BytesIO(audio_recording)
    audio_file.name = "audio.wav"  # Whisper requires a filename with a valid extension

    # model_size = "medium"
    model_size = "large-v3-turbo"

    if torch.cuda.is_available():
        DEVICE = "cuda"  # CUDA for NVIDIA GPUs
    elif torch.backends.mps.is_available():
        DEVICE = "cpu"  # MPS for Apple Silicon (M1/M2/M3)
    else:
        DEVICE = "cpu"  # Fallback to CPU
    model = WhisperModel(model_size, device=DEVICE, compute_type="int8")

    segments, info = model.transcribe(audio_file, beam_size=5)
    segments = list(segments)
    transcription = [segment.text for segment in segments]
    print(f"segments: {segments}, openai transcription: {transcription}")
    return transcription


# def transcribe_preview(session):
#     if session["audio_buffer"] is not None:
#         text = transcribe_whisper(session["audio_buffer"])
#         # send transcription
#         ws = session.get("websocket")
#         if ws:
#             message = {
#                 "event": "recognizing",
#                 "text": text,
#                 "language": session["language"]
#             }
#             ws.send(json.dumps(message))


def predict_personality(text):
    scentences = re.split("(?<=[.!?]) +", text)
    text_vector_31 = vectorizer_31.transform(scentences)
    text_vector_30 = vectorizer_30.transform(scentences)
    EXT = cEXT.predict(text_vector_31)
    NEU = cNEU.predict(text_vector_30)
    AGR = cAGR.predict(text_vector_31)
    CON = cCON.predict(text_vector_31)
    OPN = cOPN.predict(text_vector_31)
    return [EXT[0], NEU[0], AGR[0], CON[0], OPN[0]]   

@app.route("/chats/<chat_session_id>/sessions", methods=["POST"])
def open_session(chat_session_id):
    """Open a new voice input session and start continuous recognition.
    ---
    tags:
      - Sessions
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: ID of the chat session
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - language
          properties:
            language:
              type: string
              description: Language code for speech recognition (e.g., en-US)
    responses:
      200:
        description: Session created successfully
        schema:
          type: object
          properties:
            session_id:
              type: string
              description: Unique identifier for the voice recognition session
      400:
        description: Language parameter missing
        schema:
          type: object
          properties:
            error:
              type: string
              description: Description of the error
    """
    session_id = str(uuid.uuid4())

    body = request.get_json()
    if "language" not in body:
        return jsonify({"error": "Language not specified"}), 400
    language = body["language"]

    sessions[session_id] = {
        "audio_buffer": None,
        "chatSessionId": chat_session_id,
        "language": language,
        "websocket": None,  # will be set when the client connects via WS
    }

    return jsonify({"session_id": session_id})


@app.route("/chats/<chat_session_id>/sessions/<session_id>/wav", methods=["POST"])
def upload_audio_chunk(chat_session_id, session_id):
    """Upload an audio chunk (expected 16kb, ~0.5s of WAV data).
    The chunk is appended to the push stream for the session.
    ---
    tags:
      - Sessions
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: ID of the chat session
      - name: session_id
        in: path
        type: string
        required: true
        description: ID of the voice input session
      - name: audio_chunk
        in: body
        required: true
        schema:
          type: string
          format: binary
          description: Raw WAV audio data
    responses:
      200:
        description: Audio chunk received successfully
        schema:
          type: object
          properties:
            status:
              type: string
              description: Status message
      404:
        description: Session not found
        schema:
          type: object
          properties:
            error:
              type: string
              description: Description of the error
    """
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    audio_data = request.get_data()  # raw binary data from the POST body

    if sessions[session_id]["audio_buffer"] is not None:
        sessions[session_id]["audio_buffer"] += audio_data
    else:
        sessions[session_id]["audio_buffer"] = audio_data

    # TODO optionally transcribe real time audio chunks, see transcribe_preview()

    return jsonify({"status": "audio_chunk_received"})


@app.route("/chats/<chat_session_id>/sessions/<session_id>", methods=["DELETE"])
def close_session(chat_session_id, session_id):
    """Close the session (stop recognition, close push stream, cleanup).

    ---
    tags:
      - Sessions
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: The ID of the chat session
      - name: session_id
        in: path
        type: string
        required: true
        description: The ID of the session to close
    responses:
      200:
        description: Session successfully closed
        schema:
          type: object
          properties:
            status:
              type: string
              example: session_closed
      404:
        description: Session not found
        schema:
          type: object
          properties:
            error:
              type: string
              example: Session not found
    """
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    if sessions[session_id]["audio_buffer"] is not None:
        # TODO preprocess audio/text, extract and save speaker identification
        text = transcribe_whisper(sessions[session_id]["audio_buffer"])

        # Active Personality Trait Prediction
        sentiment_text = text  # 'It is important to note that each of the five personality factors represents a range'
        predictions = predict_personality(sentiment_text)
        print("predicted personality:", predictions)
        df = pd.DataFrame(dict(r=predictions, theta=["EXT", "NEU", "AGR", "CON", "OPN"]))

        #Generate Output Stream using Gemma3:1b
        stream = chat(
          model='gemma3:1b',
          messages=[{'role': 'user', 'content': 'Answer this asked by user'
          +str(text)+
          'Give reply based on personality traits without mentioning about it in response'
          +str(df.to_string())}],
          stream=True,)

        for chunk in stream:
          print(chunk['message']['content'], end='', flush=True)

        # send transcription
        ws = sessions[session_id].get("websocket")
        if ws:
            message = {
                "event": "recognized",
                "text": text,
                "personality_traits": df,
                "response_given": str(chunk['message']['content'] for chunk in stream),
                "language": sessions[session_id]["language"],
            }
            ws.send(json.dumps(message))

    # # Remove from session store
    sessions.pop(session_id, None)

    return jsonify({"status": "session_closed"})


@sock.route("/ws/chats/<chat_session_id>/sessions/<session_id>")
def speech_socket(ws, chat_session_id, session_id):
    """WebSocket endpoint for clients to receive STT results.

    This WebSocket allows clients to connect and receive speech-to-text (STT) results
    in real time. The connection is maintained until the client disconnects. If the
    session ID is invalid, an error message is sent, and the connection is closed.

    ---
    tags:
      - Sessions
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: The unique identifier for the chat session.
      - name: session_id
        in: path
        type: string
        required: true
        description: The unique identifier for the speech session.
    responses:
      400:
        description: Session not found.
      101:
        description: WebSocket connection established.
    """
    if session_id not in sessions:
        ws.send(json.dumps({"error": "Session not found"}))
        return

    # Store the websocket reference in the session
    sessions[session_id]["websocket"] = ws

    # Keep the socket open to send events
    # Typically we'd read messages from the client in a loop if needed
    while True:
        # If the client closes the socket, an exception is thrown or `ws.receive()` returns None
        msg = ws.receive()
        if msg is None:
            break


@app.route("/chats/<chat_session_id>/set-memories", methods=["POST"])
def set_memories(chat_session_id):
    """Set memories for a specific chat session by storing chat messages with embeddings.
    ---
    tags:
      - Memories
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: The unique identifier of the chat session.
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            chat_history:
              type: array
              items:
                type: object
                properties:
                  text:
                    type: string
                    description: The chat message text.
              description: List of chat messages in the session.
    responses:
      200:
        description: Memories stored successfully.
        schema:
          type: object
          properties:
            success:
              type: string
              example: "1"
      400:
        description: Invalid request data.
    """
    print(">>> set-memories called")
    data = request.get_json()
    if not data or "chat_history" not in data:
        return jsonify({"error": "Invalid data, chat_history missing"}), 400

    chat_history = data["chat_history"]

    # Preprocess and store each chat message with its embedding.
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    for message in chat_history:
        text = message.get("text", "")
        if text.strip() == "":
            continue
        embedding = get_embedding(text)
        # Store embedding as JSON string
        embedding_str = json.dumps(embedding)
        c.execute(
            "INSERT INTO memories (chat_session_id, text, embedding) VALUES (?, ?, ?)",
            (chat_session_id, text, embedding_str),
        )
    conn.commit()
    conn.close()

    print(f"{chat_session_id}: Stored {len(chat_history)} memories.")
    return jsonify({"success": "1"})


@app.route("/chats/<chat_session_id>/get-memories", methods=["GET"])
def get_memories(chat_session_id):
    """Retrieve stored memories for a specific chat session using vector similarity.
    Optionally, a 'query' parameter can be passed to retrieve the most relevant memories.
    ---
    tags:
      - Memories
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: The unique identifier of the chat session.
      - name: query
        in: query
        type: string
        required: false
        description: Optional query text to retrieve relevant memories.
    responses:
      200:
        description: Successfully retrieved memories for the chat session.
        schema:
          type: object
          properties:
            memories:
              type: array
              items:
                type: object
                properties:
                  text:
                    type: string
                  similarity:
                    type: number
      400:
        description: Invalid chat session ID.
      404:
        description: Chat session not found.
    """
    print(">>> get-memories called")
    query_text = request.args.get("query", None)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT text, embedding FROM memories WHERE chat_session_id = ?", (chat_session_id,))
    rows = c.fetchall()
    conn.close()

    if not rows:
        return jsonify({"memories": []})

    memories = []
    if query_text:
        # Compute embedding for the query text
        query_embedding = get_embedding(query_text)
        # Calculate cosine similarity with each memory
        for text, emb_str in rows:
            emb = json.loads(emb_str)
            similarity = cosine_similarity(query_embedding, emb)
            memories.append({"text": text, "similarity": similarity})
        # Sort memories by similarity (descending) and return top 3
        memories = sorted(memories, key=lambda x: x["similarity"], reverse=True)[:3]
    else:
        # If no query is provided, return all memories without similarity score.
        memories = [{"text": row[0]} for row in rows]

    print(f"{chat_session_id}: Retrieved {len(memories)} memories.")
    return jsonify({"memories": memories})


if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)
