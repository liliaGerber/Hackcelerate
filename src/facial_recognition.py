import cv2
import mediapipe as mp
import numpy as np
import time
import os
from insightface.app import FaceAnalysis

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, refine_landmarks=True)

face_recognizer = FaceAnalysis(providers=['CPUExecutionProvider'])
face_recognizer.prepare(ctx_id=0, det_size=(640, 640))

KNOWN_FACES = [
    ("Alice", "Data/person1.jpg"),
    ("Bob", "Data/person2.jpg")]

known_faces = {}
for name, img_path in KNOWN_FACES:
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARNING] Could not read image: {img_path}. Check the file path.")
        continue
    faces = face_recognizer.get(img)
    if faces:
        known_faces[name] = faces[0].normed_embedding
    else:
        print(f"[WARNING] No face detected in image: {img_path}.")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

lip_movement_threshold = 0.01
previous_lip_distance = {}
speaking_status = {}
last_speaking_time = {}
silent_threshold = 1.5


def calculate_lip_distance(landmarks):y
    return abs(landmarks[13].y - landmarks[14].y)


def get_face_identity(face_embedding, threshold=0.5):
    return max(known_faces, key=lambda name: np.dot(face_embedding, known_faces[name]), default=None)


prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Parallel face recognition and facial landmark detection
    face_rec_results = face_recognizer.get(frame)
    results = face_mesh.process(rgb_frame)

    detected_faces = {}
    for idx, face in enumerate(face_rec_results):
        name = get_face_identity(face.normed_embedding) or f"Person_{len(known_faces) + 1}"
        detected_faces[idx] = name

    active_speakers = []
    if results and results.multi_face_landmarks:
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
        print(f"Currently Speaking: {active_speakers}")

    current_time = time.time()
    print(f"FPS: {1 / (current_time - prev_time):.2f}")
    prev_time = current_time

cap.release()
