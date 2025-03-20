import io
import json
import logging
import os
import pickle
import re
import sqlite3
import threading
import time
import uuid
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import simple_websocket
import spacy
import torch
from flasgger import Swagger
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sock import Sock
from insightface.app import FaceAnalysis
from ollama import chat
from sentence_transformers import SentenceTransformer
from spacy.matcher import Matcher
from sqlalchemy import Column, Integer, Numeric, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
import pyttsx3

cv2.ocl.setUseOpenCL(True)

# ----------------------- Flask App Setup -----------------------
print("[INFO] Initializing Flask app and extensions...")
app = Flask(__name__)
sock = Sock(app)
cors = CORS(app)
swagger = Swagger(app)

# Global session store and client set
sessions = {}
clients: set[simple_websocket.ws.Server] = set()
# Both raw SQLite and SQLAlchemy will use the same database file.
DB_PATH = os.path.join(os.path.dirname(__file__), "memories.sqlite")
current_speaker = None

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
print("[INFO] Flask app and logging configured.")

# ----------------------- Model and Vectorizer Loading -----------------------
def load_pickle_model(path):
    print(f"[INFO] Loading pickle model from {path}...")
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"[INFO] Loaded pickle model from {path}.")
    return model

cEXT = load_pickle_model("data/models/cEXT.p")
cNEU = load_pickle_model("data/models/cNEU.p")
cAGR = load_pickle_model("data/models/cAGR.p")
cCON = load_pickle_model("data/models/cCON.p")
cOPN = load_pickle_model("data/models/cOPN.p")
vectorizer_31 = load_pickle_model("data/models/vectorizer_31.p")
vectorizer_30 = load_pickle_model("data/models/vectorizer_30.p")

print("[INFO] Loading SentenceTransformer model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("[INFO] SentenceTransformer model loaded.")

# ----------------------- Raw SQLite Database Setup -----------------------
def init_memories_table():
    """
    Initialize the memories table using raw SQLite.
    This will create the memories table if it does not already exist.
    """
    print("[INFO] Initializing SQLite memories table...")
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                chat_session_id TEXT,
                text TEXT,
                embedding TEXT,
                entity TEXT,
                current_summary TEXT
            )
        """)
        conn.commit()
    print("[INFO] Memories table is ready.")

# ----------------------- SQLAlchemy Database Setup -----------------------
print("[INFO] Setting up SQLAlchemy and facial recognition database...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "memories.sqlite")
print("[INFO] Database path: ", DATABASE_PATH)
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"
Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)
db_session = Session()
nlp = spacy.load("en_core_web_sm")
print("[INFO] SQLAlchemy engine and spaCy loaded.")

# The User table is assumed to already exist in the database.
class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True, autoincrement=True)
    openness = Column(Numeric, nullable=True)
    conscientiousness = Column(Numeric, nullable=True)
    extraversion = Column(Numeric, nullable=True)
    agreeableness = Column(Numeric, nullable=True)
    neuroticism = Column(Numeric, nullable=True)
    name = Column(String, nullable=True)
    nickname = Column(String, nullable=True)
    image = Column(String, nullable=True)
    preferences = Column(Text)

# We call create_all so that SQLAlchemy can use the table if it exists.
# Since the user table is assumed to exist, this won't overwrite it.
Base.metadata.create_all(engine)
print("[INFO] SQLAlchemy tables are ready.")

# ----------------------- Facial Recognition Initialization -----------------------
print("[INFO] Initializing facial recognition components...")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, refine_landmarks=True)
face_recognizer = FaceAnalysis(providers=["CPUExecutionProvider"])
face_recognizer.prepare(ctx_id=0, det_size=(640, 640))
print("[INFO] Face recognizer prepared.")

# Load known faces from the database (images stored in "Data" folder)
known_faces = {}
users = db_session.query(User).all()
print(f"[INFO] Found {len(users)} users in the database.")
for user in users:
    if user.image:
        img_path = os.path.join(os.path.dirname(__file__), "Data", user.image)
        print(f"[INFO] Checking image for user {user.name} at {img_path}...")
        if os.path.exists(img_path):
            print(f"[INFO] Loading image: {img_path}")
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = face_recognizer.get(img_rgb)
            if faces:
                known_faces[user.name] = faces[0].normed_embedding  # Embedding assumed normalized
                print(f"[INFO] Face detected and stored for {user.name}.")
            else:
                print(f"[WARNING] No face detected in image: {img_path}.")
        else:
            print(f"[WARNING] Image path does not exist: {img_path}.")
print("[INFO] Facial recognition initialization complete.")

# Parameters for lip movement / speaking detection
lip_movement_threshold = 0.01
previous_lip_distance = {}
speaking_status = {}
last_speaking_time = {}
silent_threshold = 1.5
print("[INFO] Facial recognition parameters set.")

def calculate_lip_distance(landmarks):
    """Calculate the vertical distance between two key lip landmarks."""
    distance = abs(landmarks[13].y - landmarks[14].y)
    return distance

def get_face_identity(face_embedding, threshold=0.6):
    """Return the identity of the face if similarity exceeds threshold."""
    if not known_faces:
        return None
    similarities = {name: np.dot(face_embedding, embedding) for name, embedding in known_faces.items()}
    best_match = max(similarities, key=similarities.get)
    if similarities[best_match] >= threshold:
        # print(f"[INFO] Recognized face as: {best_match}")
        return best_match
    return None

# ----------------------- Face Recognition Loop (Active Speaker Logic) -----------------------
def face_recognition_loop(stop_event):
    global current_speaker
    print("[INFO] Starting face recognition loop...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print("[ERROR] Camera not accessible.")
        return

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to read frame from camera.")
            break

        scale_factor = 0.5
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_rec_results = face_recognizer.get(frame)
        results = face_mesh.process(rgb_small_frame)

        detected_faces = {}
        if face_rec_results:
            for face in face_rec_results:
                identity = get_face_identity(face.normed_embedding) or f"Person_{len(known_faces) + 1}"
                best_match_id = None
                min_distance = float("inf")
                if results.multi_face_landmarks:
                    for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
                        nose = face_landmarks.landmark[1]
                        landmark_center = np.array(
                            [nose.x * frame.shape[1] * scale_factor, nose.y * frame.shape[0] * scale_factor]
                        )
                        face_center = np.array([face.bbox[0] + face.bbox[2] / 2, face.bbox[1] + face.bbox[3] / 2])
                        distance = np.linalg.norm(face_center - landmark_center)
                        if distance < min_distance:
                            min_distance = distance
                            best_match_id = face_id
                if best_match_id is not None:
                    detected_faces[best_match_id] = identity
            # print(f"[DEBUG] Detected faces: {detected_faces}")

        active_speakers_local = []
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
                    speaker = detected_faces.get(face_id, "Unknown")
                    active_speakers_local.append(speaker)
                    speaking_status[face_id] = True
                    last_speaking_time[face_id] = time.time()
                    # print(f"[INFO] Detected speaking from face_id {face_id}: {speaker}")
                elif not is_speaking and speaking_status[face_id]:
                    if time.time() - last_speaking_time[face_id] > silent_threshold:
                        speaking_status[face_id] = False
                        # print(f"[INFO] Face_id {face_id} is now silent.")
                previous_lip_distance[face_id] = lip_distance

        if active_speakers_local:
            current_speaker = active_speakers_local[0]
            # print("[INFO] Currently Speaking:", current_speaker)
        else:
            current_speaker = None
            # print("[DEBUG] No active speakers detected.")

    print("[INFO] Exiting face recognition loop...")
    cap.release()

# ----------------------- Additional Utility Functions -----------------------
def get_embedding(text):
    print(f"[DEBUG] Generating embedding for text: {text[:30]}...")
    embedding = embedding_model.encode(text)
    print("[DEBUG] Embedding generated.")
    return embedding.tolist()

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    similarity = float(np.dot(vec1, vec2) / (norm1 * norm2))
    print(f"[DEBUG] Calculated cosine similarity: {similarity:.4f}")
    return similarity

def transcribe_whisper(audio_recording: bytes, pipe):
    print("[INFO] Starting transcription using Whisper...")
    audio_file = io.BytesIO(audio_recording)
    audio_file.name = "audio.wav"
    outputs = pipe(
        audio_recording,
        chunk_length_s=10,
        batch_size=24,
        return_timestamps=False
    )
    transcription = outputs["text"]
    print(f"[INFO] Transcription segments: {transcription}")
    return transcription

def predict_personality(text: str) -> list[np.int32]:
    print(f"[INFO] Predicting personality for text: {text[:50]}...")
    sentences = re.split(r"(?<=[.!?]) +", text)
    text_vector_31 = vectorizer_31.transform(sentences)
    text_vector_30 = vectorizer_30.transform(sentences)
    ext = cEXT.predict(text_vector_31)
    neu = cNEU.predict(text_vector_30)
    agr = cAGR.predict(text_vector_31)
    con = cCON.predict(text_vector_31)
    opn = cOPN.predict(text_vector_31)
    predictions = [ext[0], neu[0], agr[0], con[0], opn[0]]
    print(f"[INFO] Predicted personality traits: {predictions}")
    return predictions

# ----------------------- Flask Endpoints -----------------------
@sock.route("/ws")
def websocket(ws):
    print("[INFO] WebSocket connection established.")
    clients.add(ws)
    try:
        while True:
            message = ws.receive()
            print(f"[DEBUG] Received WebSocket message: {message}")
            if message is None:
                break
    finally:
        clients.remove(ws)
        print("[INFO] WebSocket connection closed.")

@app.route("/chats/<chat_session_id>/sessions", methods=["POST"])
def open_session(chat_session_id):
    print(f"[INFO] Opening session for chat_session_id: {chat_session_id}")
    session_id = str(uuid.uuid4())
    body = request.get_json()
    if "language" not in body:
        print("[ERROR] Language not specified in session open request.")
        return jsonify({"error": "Language not specified"}), 400
    stop_event = threading.Event()
    face_thread = threading.Thread(target=face_recognition_loop, args=(stop_event,), daemon=True)
    face_thread.start()
    sessions[session_id] = {
        "audio_buffer": None,
        "chatSessionId": chat_session_id,
        "language": body["language"],
        "websocket": None,
        "face_stop_event": stop_event,
        "face_thread": face_thread,
    }
    print(f"[INFO] Session opened with session_id: {session_id}")
    return jsonify({"session_id": session_id})

@app.route("/chats/<chat_session_id>/sessions/<session_id>/wav", methods=["POST"])
def upload_audio_chunk(chat_session_id, session_id):
    print(f"[INFO] Uploading audio chunk for session_id: {session_id}")
    if session_id not in sessions:
        print("[ERROR] Session not found for audio upload.")
        return jsonify({"error": "Session not found"}), 404

    audio_data = request.get_data()
    if sessions[session_id]["audio_buffer"] is not None:
        sessions[session_id]["audio_buffer"] += audio_data
    else:
        sessions[session_id]["audio_buffer"] = audio_data
    print(f"[INFO] Audio chunk received for session_id: {session_id}")
    return jsonify({"status": "audio_chunk_received"})

@app.route("/chats/<chat_session_id>/sessions/<session_id>", methods=["DELETE"])
def close_session(chat_session_id, session_id):
    print(f"[INFO] Closing session with session_id: {session_id}")
    if session_id not in sessions:
        print("[ERROR] Session not found during close.")
        return jsonify({"error": "Session not found"}), 404

    session = sessions[session_id]
    if session["audio_buffer"] is not None:
        print("[INFO] Processing audio buffer for transcription...")
        model_size = "large-v3-turbo"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"[INFO] Using device: {device} for transcription.")

        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-" + str(model_size),
            torch_dtype=torch.float16,
            device=device,
            model_kwargs={"attn_implementation": "flash_attention_2"}
            if is_flash_attn_2_available()
            else {"attn_implementation": "sdpa"},
        )
        transcription = transcribe_whisper(session["audio_buffer"], pipe)
        text = (str(*transcription) if isinstance(transcription, list) else str(transcription)).strip()
        print(f"[INFO] Final transcription: {text}")
        predictions = predict_personality(text)
        df = pd.DataFrame({"r": predictions, "theta": ["EXT", "NEU", "AGR", "CON", "OPN"]})
        message_content = (
            "Answer this asked by user (No special characters) " + text +
            " Give reply based on personality traits without mentioning about it in response " +
            str(df.to_string())
        )
        print("[INFO] Sending message to chat model with content:")
        print(message_content)
        stream = chat(
            model="gemma3:1b",
            messages=[{"role": "user", "content": message_content}],
            stream=True,
        )

        response_content = ""
        print("[INFO] Streaming response from chat model...")
        for chunk in stream:
            chunk_text = chunk["message"]["content"]
            print(f"[DEBUG] Chat model chunk: {chunk_text}", end="", flush=True)
            response_content += chunk_text
        print("\n[INFO] Completed streaming chat model response.")

        print("[INFO] Converting response text to speech...")
        engine = pyttsx3.init()
        engine.setProperty('rate', 200)
        engine.setProperty('volume', 0.8)
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id)
        engine.say(response_content)
        engine.runAndWait()
        print("[INFO] Response has been spoken.")

        ws = session.get("websocket")
        if ws:
            message = {
                "event": "recognized",
                "text": text,
                "language": session["language"],
            }
            print("[INFO] Sending recognized event to websocket...")
            ws.send(json.dumps(message))

        for client in clients:
            try:
                data: list[Any] = [prediction.item() for prediction in predictions]
                data.append("John")  # TODO: hardcoded customer name for now
                data.append(response_content)
                data.append(text)
                print(f"[INFO] Sending data to client: {data}")
                client.send(json.dumps(data))
            except Exception as e:
                print(f"[ERROR] Failed to send data to a client: {e}")
                pass
    else:
        print("[INFO] No audio buffer to process.")

    session["face_stop_event"].set()
    session["face_thread"].join()
    print("[INFO] Face recognition thread stopped.")

    global current_speaker
    current_speaker = None
    print("[INFO] Reset current_speaker to None.")

    sessions.pop(session_id, None)
    print(f"[INFO] Session {session_id} closed.")
    return jsonify({"status": "session_closed"})

@sock.route("/ws/chats/<chat_session_id>/sessions/<session_id>")

def speech_socket(ws, chat_session_id, session_id):
    print(f"[INFO] WebSocket for speech connected for session_id: {session_id}")
    if session_id not in sessions:
        ws.send(json.dumps({"error": "Session not found"}))
        print("[ERROR] Session not found for speech socket.")
        return
    sessions[session_id]["websocket"] = ws

    while True:
        msg = ws.receive()
        if msg is None:
            print("[INFO] WebSocket speech connection closed by client.")
            break
        print(f"[DEBUG] Received message on speech socket: {msg}")

def summarize_text(previous_summary, new_messages):
    new_content = " ".join([msg["text"] for msg in new_messages if msg.get("text")])
    prompt = f"Previous summary: {previous_summary}\nNew messages: {new_content}\nGenerate an updated concise summary:"
    print("[INFO] Summarizing text with prompt:")
    print(prompt)
    response = chat(model="gemma3:1b", messages=[{"role": "user", "content": prompt}])
    summary = response.get("message", {}).get("content", previous_summary)
    print(f"[INFO] Summary generated: {summary}")
    return summary

def update_user_profile(preferences_data):
    """
    Update the user record in the SQLite database for the current speaker.
    Only the 'preferences' JSON column is updated based on the recognized speaker's name.
    If no speaker is recognized, the function does nothing.
    """
    global current_speaker
    if not current_speaker or current_speaker == "Unknown":
        print("[INFO] No valid current speaker recognized; skipping update.")
        return

    print(f"[INFO] Updating user profile for current speaker: {current_speaker}")
    preferences_json = json.dumps(preferences_data)

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    query = "UPDATE user SET preferences = ? WHERE name = ?"
    cursor.execute(query, (preferences_json, current_speaker))
    conn.commit()
    conn.close()
    print("[INFO] User profile updated.")

matcher = Matcher(nlp.vocab)
_PATTERNS = [
    ("DIET_VEGAN", [{"LOWER": "vegan"}]),
    ("DIET_VEGETARIAN", [{"LOWER": "vegetarian"}]),
    ("DIET_PESCATARIAN", [{"LOWER": "pescatarian"}]),
    ("DIET_OMNIVORE", [{"LOWER": "omnivore"}]),
    ("ALLERGY_NUTS", [{"LOWER": "nuts"}]),
    ("ALLERGY_DAIRY", [{"LOWER": "dairy"}]),
    ("ALLERGY_GLUTEN", [{"LOWER": "gluten"}]),
    ("ALLERGY_SHELLFISH", [{"LOWER": "shellfish"}])
]
for label, pattern in _PATTERNS:
    matcher.add(label, [pattern])
print("[INFO] Matcher for diet/allergy extraction initialized.")

def process_chat_history(chat_text):
    """
    Process the chat text using spaCy’s NLP pipeline to extract user preferences.

    This implementation assumes that your spaCy model has been trained (or extended)
    to recognize custom entity labels:
      - "DIET" for dietary preferences (e.g., "omnivore", "vegan")
      - "ALLERGY" for allergy mentions (e.g., "nuts", "gluten")
      - "FOOD" for food items the user likes
      - "DRINK" for beverage preferences
      - "HOBBY" for hobbies the user enjoys

    The function returns a dictionary with the extracted preferences.
    """
    print("[INFO] Processing chat history...")
    doc = nlp(chat_text)
    preferences = {}

    for ent in doc.ents:
        label = ent.label_
        text = ent.text.strip().lower()
        if label == "DIET":
            preferences["diet"] = text
            print(f"[DEBUG] Extracted DIET: {text}")
        elif label == "ALLERGY":
            preferences.setdefault("allergies", []).append(text)
            print(f"[DEBUG] Extracted ALLERGY: {text}")
        elif label == "FOOD":
            preferences.setdefault("food_preference", []).append(text)
            print(f"[DEBUG] Extracted FOOD: {text}")
        elif label == "DRINK":
            preferences.setdefault("drink_preference", []).append(text)
            print(f"[DEBUG] Extracted DRINK: {text}")
        elif label == "HOBBY":
            preferences.setdefault("hobbies", []).append(text)
            print(f"[DEBUG] Extracted HOBBY: {text}")

    print("[INFO] Extracted preferences:", preferences)
    return preferences

def store_memories(chat_session_id, chat_history):
    global current_speaker
    print(f"[INFO] Storing memories for chat_session_id: {chat_session_id}")
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()

        all_id_rows = c.execute("SELECT id FROM memories").fetchall()
        all_ids = {row[0] for row in all_id_rows}
        current_chats = [msg for msg in chat_history if msg["id"] not in all_ids]
        print(f"[INFO] Found {len(current_chats)} new messages to store.")

        if not current_chats:
            print("[INFO] No new messages; skipping memory storage.")
            return

        last_summary_row = c.execute(
            "SELECT current_summary FROM memories WHERE chat_session_id = ? ORDER BY id DESC LIMIT 1",
            (chat_session_id,),
        ).fetchone()
        last_summary = last_summary_row[0] if last_summary_row else ""
        print(f"[INFO] Last summary: {last_summary}")

        new_summary = summarize_text(last_summary, current_chats)
        print(f"[INFO] New summary generated: {new_summary}")

        for message in current_chats:
            text = message.get("text", "").strip()
            if not text:
                continue
            embedding_str = json.dumps(get_embedding(text))
            c.execute(
                "INSERT INTO memories (id, chat_session_id, text, embedding, entity, current_summary) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (message["id"], chat_session_id, text, embedding_str, message["type"], new_summary),
            )
            print(f"[INFO] Stored message id: {message['id']}")

        conversation_text = " ".join(
            msg.get("text", "").strip() for msg in chat_history if msg.get("text", "").strip()
        )
        print(f"[INFO] Full conversation text for preference extraction: {conversation_text[:50]}...")
        preferences = process_chat_history(conversation_text)
        preferences_json = json.dumps(preferences)
        print(f"[INFO] Updating user profile with preferences: {preferences}")

        if current_speaker is not None and current_speaker != "Unknown":
            update_query = "UPDATE user SET preferences = ? WHERE name = ?"
            c.execute(update_query, (preferences_json, current_speaker))
            print(f"[INFO] Updated preferences for user: {current_speaker}")
        else:
            update_query = "UPDATE user SET preferences = ? WHERE id = ?"
            c.execute(update_query, (preferences_json, chat_session_id))
            print(f"[INFO] Updated preferences for user with id: {chat_session_id}")

        conn.commit()
        print("[INFO] Memory storage and user profile update committed.")

@app.route("/chats/<chat_session_id>/set-memories", methods=["POST"])
def set_memories(chat_session_id):
    print(f"[INFO] Received set-memories call for chat_session_id: {chat_session_id}")
    chat_history = request.get_json()
    print(f"[DEBUG] Chat history received: {chat_history}")
    threading.Thread(target=store_memories, args=(chat_session_id, chat_history)).start()
    print("[INFO] Memory storage thread started.")
    return jsonify({"success": "1"})

@app.route("/chats/<chat_session_id>/get-memories", methods=["GET"])
def get_memories(chat_session_id):
    print(f"[INFO] Retrieving memories for chat_session_id: {chat_session_id}")
    query_text = request.args.get("query", None)
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT text, embedding, entity FROM memories WHERE chat_session_id = ?", (chat_session_id,))
        rows = c.fetchall()
    print(f"[INFO] Retrieved {len(rows)} memories from database.")

    if not rows:
        return jsonify({"memories": "<empty>"})

    memories = []
    if query_text:
        print(f"[INFO] Filtering memories with query: {query_text}")
        query_embedding = get_embedding(query_text)
        for text, emb_str, _ in rows:
            emb = json.loads(emb_str)
            similarity = cosine_similarity(query_embedding, emb)
            memories.append({"text": text, "similarity": similarity})
        memories = sorted(memories, key=lambda x: x["similarity"], reverse=True)[:3]
    else:
        memories = [{"text": row[0]} for row in rows]

    combined_memories = "".join([m["text"] for m in memories])
    print(f"[INFO] Combined memories: {combined_memories[:50]}...")
    return jsonify({"memories": combined_memories})

# ----------------------- Application Startup -----------------------
if __name__ == "__main__":
    print("[INFO] Starting application...")
    init_memories_table()  # Initialize memories table using raw SQLite.
    # Note: The user table is assumed to already exist.
    app.run(debug=True, host="0.0.0.0", port=5000)
