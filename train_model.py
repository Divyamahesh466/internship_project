import cv2
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import pickle

# ----------------------------
# Load YOLO (Face Detection)
# ----------------------------
model = YOLO("yolov8n.pt")  # or your custom face model

# ----------------------------
# Load ArcFace (Recognition)
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
# Cosine similarity function
# ----------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ----------------------------
# Webcam
# ----------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ----------------------------
    # YOLO Detection + ByteTrack
    # ----------------------------
    results = model.track(frame, persist=True, verbose=False)

    for r in results:
        if r.boxes is None:
            continue

        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id[0]) if box.id is not None else -1

            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            # ----------------------------
            # ArcFace recognition
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

                if best_score > 0.4:  # threshold
                    name = best_name

            # ----------------------------
            # Draw results
            # ----------------------------
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"{name} ID:{track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow("YOLO + ByteTrack + ArcFace", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()