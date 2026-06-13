import cv2
import pickle
import os
from insightface.app import FaceAnalysis

# ----------------------------
# ARCFACE MODEL
# ----------------------------
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# ----------------------------
# DATASET PATH (YOUR FOLDER NAME)
# ----------------------------
DATASET_PATH = "Data Set"

# ----------------------------
# GET EMBEDDING FUNCTION
# ----------------------------
def get_embedding(img_path):
    print("Loading:", img_path)

    img = cv2.imread(img_path)

    if img is None:
        print("❌ Image not found:", img_path)
        return None

    faces = app.get(img)

    if len(faces) == 0:
        print("❌ No face detected:", img_path)
        return None

    return faces[0].embedding

# ----------------------------
# DATABASE
# ----------------------------
db = {}

# ----------------------------
# SCAN DATASET FOLDER
# ----------------------------
for file in os.listdir(DATASET_PATH):

    if file.lower().endswith((".jpg", ".jpeg", ".png")):

        name = os.path.splitext(file)[0]   # file name = person name
        path = os.path.join(DATASET_PATH, file)

        emb = get_embedding(path)

        if emb is not None:
            db[name] = emb

# ----------------------------
# SAVE EMBEDDINGS
# ----------------------------
with open("embeddings.pkl", "wb") as f:
    pickle.dump(db, f)

print("\n✅ embeddings.pkl created successfully")

# ----------------------------
# DEBUG CHECK
# ----------------------------
for k, v in db.items():
    print(k, v.shape)