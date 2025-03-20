import time
import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Face Mesh for lip movement tracking
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,
    refine_landmarks=False
)

# Start webcam capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)

# Pre-assigned names for the first four faces
face_names = {0: "Lili", 1: "Rodrigo", 2: "Mudi", 3: "Shrey"}

# Store previous lip positions and track faces
previous_lip_distance = {}
face_trackers = {}   # face_id -> bounding box (x1, y1, x2, y2)
next_face_id = 0

# Dictionary to keep track of speaking status (face_id -> last speaking timestamp)
speaking_faces = {}
SPEAKING_TIMEOUT = 0.5  # seconds before a face is considered silent

def calculate_lip_distance(landmarks):
    """Calculates the vertical distance between the upper and lower lip."""
    return abs(landmarks[13].y - landmarks[14].y)

def find_closest_face(new_bbox):
    """Finds the closest existing face based on bounding box proximity."""
    x1_new, y1_new, x2_new, y2_new = new_bbox
    min_distance = float("inf")
    closest_face_id = None
    for face_id, (x1_old, y1_old, x2_old, y2_old) in face_trackers.items():
        distance = np.linalg.norm(np.array([x1_new, y1_new]) - np.array([x1_old, y1_old]))
        if distance < min_distance and distance < 50:  # Adjust threshold if needed
            min_distance = distance
            closest_face_id = face_id
    return closest_face_id

prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape
    results = face_mesh.process(rgb_frame)

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    updated_face_trackers = {}

    if results and results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Create bounding box for the detected face
            x_min = int(min(lm.x for lm in face_landmarks.landmark) * w)
            y_min = int(min(lm.y for lm in face_landmarks.landmark) * h)
            x_max = int(max(lm.x for lm in face_landmarks.landmark) * w)
            y_max = int(max(lm.y for lm in face_landmarks.landmark) * h)
            new_bbox = (x_min, y_min, x_max, y_max)

            # Assign or match a unique ID to the detected face
            face_id = find_closest_face(new_bbox)
            if face_id is None:
                face_id = next_face_id
                next_face_id += 1

            updated_face_trackers[face_id] = new_bbox  # Update tracking

            # Detect lip movement to determine if the face is speaking
            lip_distance = calculate_lip_distance(face_landmarks.landmark)
            if face_id in previous_lip_distance:
                movement = abs(lip_distance - previous_lip_distance[face_id])
                if movement > 0.01:  # lip movement threshold
                    speaking_faces[face_id] = time.time()  # Update last speaking time
            previous_lip_distance[face_id] = lip_distance

    # Remove faces that haven't spoken recently
    speaking_faces = {
        face_id: last_time for face_id, last_time in speaking_faces.items()
        if time.time() - last_time <= SPEAKING_TIMEOUT
    }

    # Draw bounding boxes and labels for each tracked face
    for face_id, (x1, y1, x2, y2) in face_trackers.items():
        color = (0, 0, 255) if face_id in speaking_faces else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Use the pre-assigned name if available, else use a generic label
        name = face_names.get(face_id, f"Face {face_id}")
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2, cv2.LINE_AA)

    # Draw lip landmarks for visual reference
    lip_indices = [61, 185, 40, 39, 37, 267, 269, 270, 409, 78, 191, 80, 81, 82,
                   13, 312, 311, 310, 415, 146, 91, 181, 84, 17, 314, 405, 321,
                   375, 178, 87, 14, 317, 402]
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx in lip_indices:
                if idx < len(face_landmarks.landmark):
                    lm = face_landmarks.landmark[idx]
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x, y), 2, (0, 191, 255), -1)

    # Display speaking faces in the bottom-left corner
    if speaking_faces:
        speaking_text = "Speaking: " + ", ".join(face_names.get(fid, f"Face {fid}") for fid in speaking_faces.keys())
        cv2.putText(frame, speaking_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Display FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.namedWindow("Face Tracking", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Face Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # Show the frame with overlays
    cv2.imshow("Face Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    face_trackers = updated_face_trackers.copy()

cap.release()
cv2.destroyAllWindows()
