import cv2
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import pickle

# ----------------------------
# YOLO MODEL (Detection + Tracking)
# ----------------------------
model = YOLO("yolov8n.pt")  
# NOTE: YOLO already supports ByteTrack internally via track()

# ----------------------------
# ArcFace (Recognition)
# ----------------------------
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# ----------------------------
# Load embeddings
# ----------------------------
with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)

known_embeddings = data["embeddings"]
known_names = data["names"]

# ----------------------------
# Cosine similarity
# ----------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ----------------------------
# Webcam
# ----------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ----------------------------
    # YOLO + BYTE TRACK
    # ----------------------------
    results = model.track(frame, persist=True, verbose=False)

    for r in results:
        if r.boxes is None:
            continue

        boxes = r.boxes

        for box in boxes:

            # bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # TRACK ID (ByteTrack)
            track_id = int(box.id[0]) if box.id is not None else -1

            # crop face region
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            # ----------------------------
            # ARCFACE RECOGNITION
            # ----------------------------
            faces = app.get(face_crop)

            name = "Unknown"

            if len(faces) > 0:
                emb = faces[0].embedding

                best_score = -1
                best_name = "Unknown"

                for i, known_emb in enumerate(known_embeddings):
                    score = cosine_similarity(emb, known_emb)

                    if score > best_score:
                        best_score = score
                        best_name = known_names[i]

                if best_score > 0.4:
                    name = best_name

            # ----------------------------
            # DRAW OUTPUT
            # ----------------------------
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"{name} ID:{track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    cv2.imshow("YOLO + BYTE TRACK + ARCFACE", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()