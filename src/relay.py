import io
import json
import pickle
import re
import sqlite3
import uuid
from typing import Any

import numpy as np
import pandas as pd
import simple_websocket
import torch
from faster_whisper import WhisperModel
from flasgger import Swagger
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sock import Sock
from ollama import chat
from sentence_transformers import SentenceTransformer

# Flask app and extensions
app = Flask(__name__)
sock = Sock(app)
cors = CORS(app)
swagger = Swagger(app)

# Global session store and constants
sessions = {}
clients: set[simple_websocket.ws.Server] = set()
DB_PATH = "memories.sqlite"


# Load models and vectorizers using context managers
def load_pickle_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


cEXT = load_pickle_model("data/models/cEXT.p")
cNEU = load_pickle_model("data/models/cNEU.p")
cAGR = load_pickle_model("data/models/cAGR.p")
cCON = load_pickle_model("data/models/cCON.p")
cOPN = load_pickle_model("data/models/cOPN.p")
vectorizer_31 = load_pickle_model("data/models/vectorizer_31.p")
vectorizer_30 = load_pickle_model("data/models/vectorizer_30.p")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# ----------------------- Database Setup -----------------------


def init_db():
    """Initialize the SQLite database and create the memories table if it doesn't exist."""
    with sqlite3.connect(DB_PATH) as conn:
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


# ----------------------- Utility Functions -----------------------


def get_embedding(text):
    """Compute embedding for the given text using SentenceTransformer model."""
    embedding = embedding_model.encode(text)
    return embedding.tolist()


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def transcribe_whisper(audio_recording):
    """Transcribe audio recording using Whisper.
    Returns a list of transcription segments.
    """
    audio_file = io.BytesIO(audio_recording)
    audio_file.name = "audio.wav"  # Whisper requires a valid filename extension

    model_size = "large-v3-turbo"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "cpu"  # Use CPU for MPS devices as needed
    else:
        device = "cpu"
    model = WhisperModel(model_size, device=device, compute_type="int8")

    segments, info = model.transcribe(audio_file, beam_size=5)
    segments = list(segments)
    transcription = [segment.text for segment in segments]
    print(f"Transcription segments: {transcription}")
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


def predict_personality(text: str) -> list[np.int32]:
    """Predict personality traits from text using pre-loaded classifiers and vectorizers.
    Returns a list: [EXT, NEU, AGR, CON, OPN].
    """
    sentences = re.split(r"(?<=[.!?]) +", text)
    text_vector_31 = vectorizer_31.transform(sentences)
    text_vector_30 = vectorizer_30.transform(sentences)
    EXT = cEXT.predict(text_vector_31)
    NEU = cNEU.predict(text_vector_30)
    AGR = cAGR.predict(text_vector_31)
    CON = cCON.predict(text_vector_31)
    OPN = cOPN.predict(text_vector_31)
    return [EXT[0], NEU[0], AGR[0], CON[0], OPN[0]]


# ----------------------- Flask Endpoints -----------------------


@sock.route("/ws")
def websocket(ws):
    clients.add(ws)
    try:
        while True:
            message = ws.receive()
            if message is None:
                break
    finally:
        clients.remove(ws)


@app.route("/chats/<chat_session_id>/sessions", methods=["POST"])
def open_session(chat_session_id):
    """Open a new voice input session and start continuous recognition."""
    session_id = str(uuid.uuid4())
    body = request.get_json()
    if "language" not in body:
        return jsonify({"error": "Language not specified"}), 400

    sessions[session_id] = {
        "audio_buffer": None,
        "chatSessionId": chat_session_id,
        "language": body["language"],
        "websocket": None,  # will be set when the client connects via WS
    }
    return jsonify({"session_id": session_id})


@app.route("/chats/<chat_session_id>/sessions/<session_id>/wav", methods=["POST"])
def upload_audio_chunk(chat_session_id, session_id):
    """Upload an audio chunk and append to the session's buffer."""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    audio_data = request.get_data()  # Raw binary data from the POST body
    if sessions[session_id]["audio_buffer"] is not None:
        sessions[session_id]["audio_buffer"] += audio_data
    else:
        sessions[session_id]["audio_buffer"] = audio_data

    # TODO: Optionally, transcribe real-time audio chunks (see transcribe_preview)

    return jsonify({"status": "audio_chunk_received"})


@app.route("/chats/<chat_session_id>/sessions/<session_id>", methods=["DELETE"])
def close_session(chat_session_id, session_id):
    """Close the session, process the audio and send a response based on personality traits."""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    session = sessions[session_id]
    if session["audio_buffer"] is not None:
        # TODO preprocess audio/text, extract and save speaker identification
        transcription = transcribe_whisper(session["audio_buffer"])
        text = str(*transcription) if isinstance(transcription, list) else str(transcription)
        predictions = predict_personality(text)
        print("Predicted personality traits:", predictions)
        df = pd.DataFrame(
            {
                "r": predictions,
                "theta": ["EXT", "NEU", "AGR", "CON", "OPN"],
            }
        )

        for client in clients:
            try:
                data: list[Any] = [prediction.item() for prediction in predictions]
                data.append("John")  # TODO: hardcoded customer name for now
                client.send(json.dumps(data))
            except Exception as e:
                print(e)
                pass  # Ignore errors if client disconnects

        # Generate output stream from external chat model
        message_content = (
            "Answer this asked by user "
            + text
            + " Give reply based on personality traits without mentioning about it in response "
            + str(df.to_string())
        )
        stream = chat(
            model="gemma3:1b",
            messages=[{"role": "user", "content": message_content}],
            stream=True,
        )

        response_content = ""
        for chunk in stream:
            chunk_text = chunk["message"]["content"]
            print(chunk_text, end="", flush=True)
            response_content += chunk_text

        # Send transcription and personality response via websocket if available
        ws = session.get("websocket")
        if ws:
            message = {
                "event": "recognized",
                "text": text,
                "personality_traits": df.to_dict(),
                "response_given": response_content,
                "language": session["language"],
            }
            ws.send(json.dumps(message))

    sessions.pop(session_id, None)
    return jsonify({"status": "session_closed"})


@sock.route("/ws/chats/<chat_session_id>/sessions/<session_id>")
def speech_socket(ws, chat_session_id, session_id):
    """WebSocket endpoint for clients to receive STT results.
    Maintains connection until the client disconnects.
    """
    if session_id not in sessions:
        ws.send(json.dumps({"error": "Session not found"}))
        return

    sessions[session_id]["websocket"] = ws

    while True:
        msg = ws.receive()
        if msg is None:
            break


@app.route("/chats/<chat_session_id>/set-memories", methods=["POST"])
def set_memories(chat_session_id):
    """Store chat messages with embeddings as memories for a given chat session."""
    data = request.get_json()
    if not data or "chat_history" not in data:
        return jsonify({"error": "Invalid data, chat_history missing"}), 400

    chat_history = data["chat_history"]
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        for message in chat_history:
            text = message.get("text", "").strip()
            if not text:
                continue
            embedding_str = json.dumps(get_embedding(text))
            c.execute(
                "INSERT INTO memories (chat_session_id, text, embedding) VALUES (?, ?, ?)",
                (chat_session_id, text, embedding_str),
            )
        conn.commit()

    print(f"{chat_session_id}: Stored {len(chat_history)} memories.")
    return jsonify({"success": "1"})


@app.route("/chats/<chat_session_id>/get-memories", methods=["GET"])
def get_memories(chat_session_id):
    """Retrieve stored memories for a specific chat session.
    Optionally filter memories based on a query parameter using cosine similarity.
    """
    query_text = request.args.get("query", None)
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT text, embedding FROM memories WHERE chat_session_id = ?", (chat_session_id,))
        rows = c.fetchall()

    if not rows:
        return jsonify({"memories": []})

    memories = []
    if query_text:
        query_embedding = get_embedding(query_text)
        for text, emb_str in rows:
            emb = json.loads(emb_str)
            similarity = cosine_similarity(query_embedding, emb)
            memories.append({"text": text, "similarity": similarity})
        memories = sorted(memories, key=lambda x: x["similarity"], reverse=True)[:3]
    else:
        memories = [{"text": row[0]} for row in rows]

    print(f"{chat_session_id}: Retrieved {len(memories)} memories.")
    return jsonify({"memories": memories})


# ----------------------- Application Startup -----------------------

if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)
