import cv2
import numpy as np
import pickle
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# ----------------------------
# YOLO MODEL (IDEALLY FACE MODEL USE PANNUNGA)
# ----------------------------
model = YOLO("yolov8n.pt")  # better: use face YOLO if available

# ----------------------------
# ARCFACE MODEL
# ----------------------------
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# ----------------------------
# LOAD EMBEDDINGS
# ----------------------------
with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)

known_embeddings = data["embeddings"]
known_names = data["names"]

# ----------------------------
# COSINE SIMILARITY
# ----------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ----------------------------
# CAMERA
# ----------------------------
cap = cv2.VideoCapture(0)

THRESHOLD = 0.4

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # ----------------------------
    # YOLO + TRACKING
    # ----------------------------
    results = model.track(frame, persist=True, verbose=False)

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # SAFE BOUNDARY FIX
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            track_id = int(box.id[0]) if box.id is not None else -1

            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            # ----------------------------
            # ARC FACE
            # ----------------------------
            faces = app.get(face_crop)

            name = "Unknown"
            best_score = -1

            if len(faces) > 0:

                emb = faces[0].embedding
                emb = emb / np.linalg.norm(emb)   # NORMALIZATION FIX

                # ----------------------------
                # MATCHING
                # ----------------------------
                for i, known_emb in enumerate(known_embeddings):

                    score = cosine_similarity(emb, known_emb)

                    if score > best_score:
                        best_score = score
                        name = known_names[i]

                if best_score < THRESHOLD:
                    name = "Unknown"

            # ----------------------------
            # DRAW BOX
            # ----------------------------
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"{name} ID:{track_id} {best_score:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

    cv2.imshow("YOLO + ByteTrack + ArcFace", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()