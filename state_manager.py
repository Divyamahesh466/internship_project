import cv2
import pickle
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# ----------------------------
# YOLO MODEL
# ----------------------------
model = YOLO("yolov8n.pt")

# ----------------------------
# ARCFACE MODEL
# ----------------------------
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# ----------------------------
# LOAD DATABASE
# ----------------------------
with open("embeddings.pkl", "rb") as f:
    db = pickle.load(f)

# ----------------------------
# COSINE SIMILARITY
# ----------------------------
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ----------------------------
# STATE MANAGER
# ----------------------------
class StateManager:
    def __init__(self):
        self.state = "OUTSIDE"

    def update(self, inside_zone):
        self.state = "INSIDE" if inside_zone else "OUTSIDE"
        return self.state

state_manager = StateManager()

# ----------------------------
# CAMERA
# ----------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # ----------------------------
    # ZONE (LEFT AREA)
    # ----------------------------
    zx1, zy1 = 0, 0
    zx2, zy2 = int(w * 0.4), h

    cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 255, 0), 2)

    # ----------------------------
    # YOLO DETECTION
    # ----------------------------
    results = model(frame)

    for r in results:
        for box in r.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            # ----------------------------
            # ARCFACE EMBEDDING
            # ----------------------------
            faces = app.get(face_crop)

            if len(faces) == 0:
                continue

            emb = faces[0].embedding

            # ----------------------------
            # MATCHING
            # ----------------------------
            name = "Unknown"
            best_score = -1

            for person, db_emb in db.items():
                score = cosine_sim(emb, db_emb)

                if score > best_score:
                    best_score = score
                    name = person

            if best_score < 0.4:
                name = "Unknown"

            # ----------------------------
            # STATE CHECK (ZONE)
            # ----------------------------
            cx = x1 + (x2 - x1) // 2
            inside_zone = cx < zx2

            state = state_manager.update(inside_zone)

            # ----------------------------
            # COLOR BASED ON STATE
            # ----------------------------
            if state == "INSIDE":
                color = (255, 0, 0)  # BLUE
            else:
                color = (0, 0, 255)  # RED

            # ----------------------------
            # DRAW BOX
            # ----------------------------
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # NAME + SCORE
            cv2.putText(
                frame,
                f"{name} {best_score:.2f}",
                (x1, y1 - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # STATE
            cv2.putText(
                frame,
                state,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    cv2.imshow("YOLO + ARCFACE + STATE SYSTEM", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()