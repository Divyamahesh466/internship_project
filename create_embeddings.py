import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis

# ----------------------------
# DATASET PATH
# ----------------------------
DATASET_PATH = "Data Set"
OUTPUT_FILE = "embeddings.pkl"

# ----------------------------
# ARCFACE MODEL (InsightFace)
# ----------------------------
app = FaceAnalysis(name="buffalo_l")

# GPU fallback (safe)
ctx_id = 0  # GPU
try:
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    print("[INFO] Using GPU")
except:
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print("[INFO] GPU not found, using CPU")

# ----------------------------
# STORAGE
# ----------------------------
embeddings = []
names = []

# ----------------------------
# CHECK DATASET
# ----------------------------
if not os.path.exists(DATASET_PATH):
    print("❌ Dataset folder not found!")
    exit()

# ----------------------------
# PROCESS DATASET
# ----------------------------
for person_folder in os.listdir(DATASET_PATH):

    person_path = os.path.join(DATASET_PATH, person_folder)

    if not os.path.isdir(person_path):
        continue

    print(f"[INFO] Processing person: {person_folder}")

    for img_name in os.listdir(person_path):

        img_path = os.path.join(person_path, img_name)

        img = cv2.imread(img_path)

        if img is None:
            print(f"❌ Cannot read image: {img_path}")
            continue

        faces = app.get(img)

        if len(faces) == 0:
            print(f"❌ No face detected: {img_path}")
            continue

        # Take first face only
        emb = faces[0].embedding

        # Normalize embedding (VERY IMPORTANT for ArcFace)
        emb = emb / np.linalg.norm(emb)

        embeddings.append(emb)
        names.append(person_folder.strip())

# ----------------------------
# SAVE DATA
# ----------------------------
data = {
    "embeddings": np.array(embeddings),
    "names": np.array(names)
}

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(data, f)

# ----------------------------
# SUMMARY
# ----------------------------
print("\n✅ embeddings.pkl created successfully!")
print("Total embeddings:", len(embeddings))
print("Total persons:", len(set(names)))