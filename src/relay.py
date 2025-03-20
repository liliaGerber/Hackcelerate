import io
import json
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
import torch
from flasgger import Swagger
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sock import Sock
from insightface.app import FaceAnalysis
from ollama import chat
from sentence_transformers import SentenceTransformer
from sqlalchemy import Column, Integer, Numeric, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available

cv2.ocl.setUseOpenCL(True)

# ----------------------- Flask App Setup -----------------------
app = Flask(__name__)
sock = Sock(app)
cors = CORS(app)
swagger = Swagger(app)

# Global session store and client set
sessions = {}
clients: set[simple_websocket.ws.Server] = set()
DB_PATH = os.path.join(os.path.dirname(__file__), "memories.sqlite")


# ----------------------- Model and Vectorizer Loading -----------------------
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
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, refine_landmarks=True)

face_recognizer = FaceAnalysis(providers=["CPUExecutionProvider"])
face_recognizer.prepare(ctx_id=0, det_size=(640, 640))

# Load known faces from the database (images stored in "Data" folder inside the same directory)
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
                known_faces[user.name] = faces[0].normed_embedding  # Embedding assumed normalized
                print(f"[INFO] Face detected for {user.name}")
            else:
                print(f"[WARNING] No face detected in image: {img_path}.")
        else:
            print(f"[WARNING] Image path does not exist: {img_path}.")
print("[INFO] known_faces:", known_faces)

# Parameters for lip movement / speaking detection
lip_movement_threshold = 0.01
previous_lip_distance = {}
speaking_status = {}
last_speaking_time = {}
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


# ----------------------- Face Recognition Loop (Active Speaker Logic) -----------------------
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

        # Run face recognition on full-resolution frame for accuracy
        face_rec_results = face_recognizer.get(frame)
        # Run face mesh on resized frame for speed
        results = face_mesh.process(rgb_small_frame)

        detected_faces = {}
        if face_rec_results:
            for face in face_rec_results:
                identity = get_face_identity(face.normed_embedding) or f"Person_{len(known_faces) + 1}"
                best_match_id = None
                min_distance = float("inf")
                if results.multi_face_landmarks:
                    for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
                        # Use the nose tip as reference
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
            print("Currently Speaking:", ", ".join(active_speakers))
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        # print(f"FPS: {fps:.2f}")
        prev_time = current_time

    cap.release()


# ----------------------- Additional Utility Functions -----------------------
def get_embedding(text):
    embedding = embedding_model.encode(text)
    return embedding.tolist()


def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def transcribe_whisper(audio_recording: bytes):
    audio_file = io.BytesIO(audio_recording)
    audio_file.name = "audio.wav"
    model_size = "large-v3-turbo"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Insanely Faster Whisper Speech to Text
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-" + str(model_size),
        torch_dtype=torch.float16,
        device=device,
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
    """Predict personality traits from text using preloaded classifiers and vectorizers.
    Returns a list: [EXT, NEU, AGR, CON, OPN].
    """
    sentences = re.split(r"(?<=[.!?]) +", text)
    text_vector_31 = vectorizer_31.transform(sentences)
    text_vector_30 = vectorizer_30.transform(sentences)
    ext = cEXT.predict(text_vector_31)
    neu = cNEU.predict(text_vector_30)
    agr = cAGR.predict(text_vector_31)
    con = cCON.predict(text_vector_31)
    opn = cOPN.predict(text_vector_31)
    return [ext[0], neu[0], agr[0], con[0], opn[0]]


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
    print(">>> call open_session")
    session_id = str(uuid.uuid4())
    body = request.get_json()
    if "language" not in body:
        return jsonify({"error": "Language not specified"}), 400
    # Start the face recognition thread (which prints active speakers)
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
    return jsonify({"session_id": session_id})


@app.route("/chats/<chat_session_id>/sessions/<session_id>/wav", methods=["POST"])
def upload_audio_chunk(chat_session_id, session_id):
    print(">>> call upload_audio_chunk")
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    audio_data = request.get_data()
    if sessions[session_id]["audio_buffer"] is not None:
        sessions[session_id]["audio_buffer"] += audio_data
    else:
        sessions[session_id]["audio_buffer"] = audio_data
    return jsonify({"status": "audio_chunk_received"})


@app.route("/chats/<chat_session_id>/sessions/<session_id>", methods=["DELETE"])
def close_session(chat_session_id, session_id):
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
        df = pd.DataFrame({"r": predictions, "theta": ["EXT", "NEU", "AGR", "CON", "OPN"]})
        message_content = (
            "Answer this asked by user "
            + text
            + " Give reply based on personality traits without mentioning about it in response "
            + str(df.to_string())
        )
        stream = chat(
            model="gemma3:1b",
            # model="deepseek-r1:1.5b"
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
                pass
    # Stop the face recognition thread
    session["face_stop_event"].set()
    session["face_thread"].join()
    sessions.pop(session_id, None)
    return jsonify({"status": "session_closed"})


@sock.route("/ws/chats/<chat_session_id>/sessions/<session_id>")
def speech_socket(ws, chat_session_id, session_id):
    print(">>> call speech_socket")
    if session_id not in sessions:
        ws.send(json.dumps({"error": "Session not found"}))
        return
    sessions[session_id]["websocket"] = ws

    while True:
        msg = ws.receive()
        if msg is None:
            break


def summarize_text(previous_summary, new_messages):
    """Generate a concise updated summary based on the previous summary and new messages."""
    new_content = " ".join([msg["text"] for msg in new_messages if msg.get("text")])
    prompt = f"Previous summary: {previous_summary}\nNew messages: {new_content}\nGenerate an updated concise summary:"
    response = chat(model="gemma3:1b", messages=[{"role": "user", "content": prompt}])
    return response.get("message", {}).get("content", previous_summary)  # Fallback to previous summary


def store_memories(chat_session_id, chat_history):
    """Store chat messages asynchronously to improve performance."""
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()

        # Fetch existing memory IDs
        all_id_rows = c.execute("SELECT id FROM memories").fetchall()
        all_ids = {row[0] for row in all_id_rows}
        current_chats = [msg for msg in chat_history if msg["id"] not in all_ids]

        if not current_chats:
            return

        # Fetch the last summary
        last_summary_row = c.execute(
            "SELECT current_summary FROM memories WHERE chat_session_id = ? ORDER BY id DESC LIMIT 1",
            (chat_session_id,),
        ).fetchone()
        last_summary = last_summary_row[0] if last_summary_row else ""

        # Generate new summary
        new_summary = summarize_text(last_summary, current_chats)

        # Insert new messages into the database
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

        conn.commit()


@app.route("/chats/<chat_session_id>/set-memories", methods=["POST"])
def set_memories(chat_session_id):
    """Store chat messages asynchronously and return success response immediately."""
    print(">>> call set_memories")
    chat_history = request.get_json()

    # background thread for memory storage
    threading.Thread(target=store_memories, args=(chat_session_id, chat_history)).start()

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
        for text, emb_str, _ in rows:
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
