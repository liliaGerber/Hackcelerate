import time
import cv2
import mediapipe as mp
import numpy as np
from insightface.app import FaceAnalysis
from sqlalchemy import create_engine, Column, Integer, String, Numeric, Text
from sqlalchemy.orm import sessionmaker, declarative_base
import os
import threading

# Enable OpenCL for potential hardware acceleration
cv2.ocl.setUseOpenCL(True)

# --- Database Initialization ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, '..', 'memories.sqlite')
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"
Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)
session = Session()

# Define the User model
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

# Ensure the table exists before querying
Base.metadata.create_all(engine)

# --- Load Known Faces from Database ---
users = session.query(User).all()
# Initialize Mediapipe FaceMesh for landmark detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, refine_landmarks=True)

# Initialize InsightFace for face recognition
face_recognizer = FaceAnalysis(providers=["CPUExecutionProvider"])
face_recognizer.prepare(ctx_id=0, det_size=(640, 640))

# Load known faces and store their embeddings in a dictionary
known_faces = {}
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

# --- Video Capture Setup ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

# Parameters for speaking detection
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
    # Since embeddings are normalized, dot product equals cosine similarity
    similarities = {name: np.dot(face_embedding, embedding) for name, embedding in known_faces.items()}
    best_match = max(similarities, key=similarities.get)
    if similarities[best_match] >= threshold:
        return best_match
    return None

prev_time = time.time()

# --- Main Processing Loop ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Optionally resize frame for faster landmark detection
    scale_factor = 0.5  # Adjust scale factor for performance vs. accuracy
    small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Run face recognition on the full-resolution frame (for best accuracy)
    face_rec_results = face_recognizer.get(frame)
    # Run face mesh on the smaller frame for speed
    results = face_mesh.process(rgb_small_frame)

    detected_faces = {}

    if face_rec_results:
        for face in face_rec_results:
            # Identify the face
            identity = get_face_identity(face.normed_embedding) or f"Person_{len(known_faces) + 1}"
            best_match_id = None
            min_distance = float('inf')

            # If face mesh detects landmarks, try to match face detection with landmarks
            if results.multi_face_landmarks:
                for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
                    # Use nose tip (landmark index 1) as reference point; adjust coordinates from small frame to original
                    nose = face_landmarks.landmark[1]
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

    # FPS calculation for performance monitoring
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    print(f"FPS: {fps:.2f}")
    prev_time = current_time

cap.release()
session.close()
