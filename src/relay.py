import io
import json
import os
import pickle
import re
import sqlite3
import threading
import time
import uuid

import cv2
import mediapipe as mp
import numpy as np
import simple_websocket
from insightface.app import FaceAnalysis
from sqlalchemy import create_engine, Column, Integer, String, Numeric, Text
from sqlalchemy.orm import sessionmaker, declarative_base

import pandas as pd
import torch
from faster_whisper import WhisperModel
from flasgger import Swagger
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sock import Sock
from ollama import chat
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
cv2.ocl.setUseOpenCL(True)

# Flask app and extensions
app = Flask(__name__)
sock = Sock(app)
cors = CORS(app)
swagger = Swagger(app)

# Global session store and constants
sessions = {}
clients: set[simple_websocket.ws.Server] = set()
DB_PATH = os.path.join(os.path.dirname(__file__), "memories.sqlite")


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
                embedding TEXT,
                entity TEXT
            )
        """)
        conn.commit()

# ----------------------- Database Setup for Facial Recognition -----------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "memories.sqlite")
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"
Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)
db_session = Session()

class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True, autoincrement=True)
    openness = Column(Numeric, nullable=True)
    conscientiousness = Column(Numeric, nullable=True)
    extraversion = Column(Numeric, nullable=True)
    agreeableness = Column(Numeric, nullable=True)
    neuroticism = Column(Numeric, nullable=True)
    food_preference = Column(Text, nullable=True)
    drink_preference = Column(Text, nullable=True)
    diet = Column(String, nullable=True)
    name = Column(String, nullable=True)
    nickname = Column(String, nullable=True)
    allergies = Column(String, nullable=True)
    hobbies = Column(Text, nullable=True)
    image = Column(String, nullable=True)

Base.metadata.create_all(engine)

# ----------------------- Facial Recognition Initialization -----------------------

# Mediapipe FaceMesh for landmark detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=5,
                                  refine_landmarks=True)

# InsightFace for face recognition
face_recognizer = FaceAnalysis(providers=["CPUExecutionProvider"])
face_recognizer.prepare(ctx_id=0, det_size=(640, 640))

# Load known faces from database
known_faces = {}
users = db_session.query(User).all()
for user in users:
    if user.image:
        img_path = os.path.join(os.path.dirname(__file__), "Data", user.image)
        if os.path.exists(img_path):
            print(f"[INFO] Loading image: {img_path}")
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = face_recognizer.get(img_rgb)
            if faces:
                known_faces[user.name] = faces[0].normed_embedding  # Assumes embedding is normalized
                print(f"[INFO] Face detected for {user.name}")
            else:
                print(f"[WARNING] No face detected in image: {img_path}.")
        else:
            print(f"[WARNING] Image path does not exist: {img_path}.")

print("[INFO] known_faces: ", known_faces)
# Parameters for lip movement / speaking detection
previous_lip_distance = {}
speaking_status = {}
last_speaking_time = {}
lip_movement_threshold = 0.01
silent_threshold = 1.5

def calculate_lip_distance(landmarks):
    """Calculate the vertical distance between two key lip landmarks."""
    return abs(landmarks[13].y - landmarks[14].y)

def get_face_identity(face_embedding, threshold=0.6):
    """Return the identity of the face if similarity exceeds threshold."""
    if not known_faces:
        return None
    similarities = {name: np.dot(face_embedding, embedding) for name, embedding in known_faces.items()}
    best_match = max(similarities, key=similarities.get)
    if similarities[best_match] >= threshold:
        return best_match
    return None

# ----------------------- Utility Functions -----------------------

def face_recognition_loop(stop_event):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print("[ERROR] Camera not accessible.")
        return

    prev_time = time.time()
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for faster face mesh detection
        scale_factor = 0.5
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Full-resolution face recognition
        face_rec_results = face_recognizer.get(frame)
        # Run face mesh on the smaller frame
        results = face_mesh.process(rgb_small_frame)

        detected_faces = {}
        if face_rec_results:
            for face in face_rec_results:
                identity = get_face_identity(face.normed_embedding) or f"Person_{len(known_faces) + 1}"
                best_match_id = None
                min_distance = float('inf')
                if results.multi_face_landmarks:
                    for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
                        nose = face_landmarks.landmark[1]  # nose tip as reference
                        landmark_center = np.array([
                            nose.x * frame.shape[1] * scale_factor,
                            nose.y * frame.shape[0] * scale_factor
                        ])
                        face_center = np.array([
                            face.bbox[0] + face.bbox[2] / 2,
                            face.bbox[1] + face.bbox[3] / 2
                        ])
                        distance = np.linalg.norm(face_center - landmark_center)
                        if distance < min_distance:
                            min_distance = distance
                            best_match_id = face_id
                if best_match_id is not None:
                    detected_faces[best_match_id] = identity

        active_speakers = []
        if results.multi_face_landmarks:
            for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
                lip_distance = calculate_lip_distance(face_landmarks.landmark)
                if face_id not in previous_lip_distance:
                    previous_lip_distance[face_id] = lip_distance
                    speaking_status[face_id] = False
                    last_speaking_time[face_id] = time.time()

                movement = abs(lip_distance - previous_lip_distance[face_id])
                is_speaking = movement > lip_movement_threshold

                if is_speaking and not speaking_status[face_id]:
                    active_speakers.append(detected_faces.get(face_id, "Unknown"))
                    speaking_status[face_id] = True
                    last_speaking_time[face_id] = time.time()
                elif not is_speaking and speaking_status[face_id]:
                    if time.time() - last_speaking_time[face_id] > silent_threshold:
                        speaking_status[face_id] = False

                previous_lip_distance[face_id] = lip_distance

        if active_speakers:
            print("Currently Speaking:", ', '.join(active_speakers))
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        # print(f"FPS: {fps:.2f}")
        prev_time = current_time

    cap.release()

# ----------------------- Model Loading for Transcription & Personality -----------------------

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


def transcribe_whisper(audio_recording: bytes):
    """Transcribe audio recording using Whisper.
    Returns a list of transcription segments.
    """
    audio_file = io.BytesIO(audio_recording)
    audio_file.name = "audio.wav"  # Whisper requires a valid filename extension

    model_size = "large-v3-turbo"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"  # Use CPU for MPS devices as needed
    else:
        device = "cpu"

    # Insanely Faster Whisper Speech to Text
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-"
        + str(model_size),  # select checkpoint from https://huggingface.co/openai/whisper-large-v3#model-details
        torch_dtype=torch.float16,
        device=device,  # or mps for Mac devices
        model_kwargs={"attn_implementation": "flash_attention_2"}
        if is_flash_attn_2_available()
        else {"attn_implementation": "sdpa"},
    )
    outputs = pipe(
        audio_recording,
        chunk_length_s=10,
        batch_size=24,
        return_timestamps=False,
    )
    transcription = outputs["text"]

    # Faster Whisper Speech to Text
    # model = WhisperModel(model_size, device=device, compute_type="int8")
    # segments, info = model.transcribe(audio_file, beam_size=5)
    # segments = list(segments)
    # transcription = [segment.text for segment in segments]

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
    print(">>> call ws")
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
    print(">>> call open_session")
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
    print(">>> call upload_audio_chunk")
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
    print(">>> call close_session")
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    session = sessions[session_id]
    if session["audio_buffer"] is not None:
        # TODO preprocess audio/text, extract and save speaker identification
        transcription = transcribe_whisper(session["audio_buffer"])
        text = (str(*transcription) if isinstance(transcription, list) else str(transcription)).strip()
        predictions = predict_personality(text)
        print("Predicted personality traits:", predictions)
        df = pd.DataFrame(
            {
                "r": predictions,
                "theta": ["EXT", "NEU", "AGR", "CON", "OPN"],
            }
        )

        # Generate output stream from external chat model
        message_content = (
            "Answer this asked by user "
            + text
            + " Give reply based on personality traits without mentioning about it in response "
            + str(df.to_string())
        )
        stream = chat(
            model="gemma3:1b",
            # model="deepseek-r1:1.5b",
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
                "language": session["language"],
            }
            ws.send(json.dumps(message))

        # TODO: Update user profile + send data to frontend2
        for client in clients:
            try:
                data: list[Any] = [prediction.item() for prediction in predictions]
                data.append("John")  # TODO: hardcoded customer name for now
                data.append(response_content)
                data.append(text)
                client.send(json.dumps(data))
            except Exception as e:
                print(e)
                pass  # Ignore errors if client disconnects

    sessions.pop(session_id, None)
    return jsonify({"status": "session_closed"})


@sock.route("/ws/chats/<chat_session_id>/sessions/<session_id>")
def speech_socket(ws, chat_session_id, session_id):
    """WebSocket endpoint for clients to receive STT results.
    Maintains connection until the client disconnects.
    """
    print(">>> call speech_socket")
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
    print(">>> call set_memories")
    chat_history = request.get_json()
    is_bot = True

    chat_history = data["chat_history"]
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        for message in chat_history:
            text = message.get("text", "").strip()
            if not text:
                continue
            embedding_str = json.dumps(get_embedding(text))
            c.execute(
                "INSERT INTO memories (chat_session_id, text, embedding, entity) VALUES (?, ?, ?, ?)",
                (chat_session_id, text, embedding_str, "BOT" if is_bot else "USER"),
            )
            is_bot = not is_bot
        conn.commit()

    print(f"{chat_session_id}: Stored {len(chat_history)} memories.")
    return jsonify({"success": "1"})


@app.route("/chats/<chat_session_id>/get-memories", methods=["GET"])
def get_memories(chat_session_id):
    """Retrieve stored memories for a specific chat session.
    Optionally filter memories based on a query parameter using cosine similarity.
    """
    print(">>> call get_memories")
    query_text = request.args.get("query", None)
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT text, embedding, entity FROM memories WHERE chat_session_id = ?", (chat_session_id,))
        rows = c.fetchall()

    if not rows:
        return jsonify({"memories": "<empty>"})

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
    return jsonify({"memories": "".join([m["text"] for m in memories])})


# ----------------------- Application Startup -----------------------

if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)
