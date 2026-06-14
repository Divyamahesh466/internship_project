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
# ARCFACE MODEL
# ----------------------------
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# ----------------------------
# STORAGE
# ----------------------------
embeddings = []
names = []

# ----------------------------
# CHECK DATASET FOLDER
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

    print(f"[INFO] Processing: {person_folder}")

    for img_name in os.listdir(person_path):

        img_path = os.path.join(person_path, img_name)

        img = cv2.imread(img_path)

        if img is None:
            print("❌ Cannot read:", img_path)
            continue

        faces = app.get(img)

        if len(faces) == 0:
            print("❌ No face:", img_path)
            continue

        emb = faces[0].embedding

        # NORMALIZE (IMPORTANT FOR ACCURACY)
        emb = emb / np.linalg.norm(emb)

        embeddings.append(emb)
        names.append(person_folder)

# ----------------------------
# SAVE PKL FILE
# ----------------------------
data = {
    "embeddings": np.array(embeddings),
    "names": names
}

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(data, f)

print("\n✅ embeddings.pkl created successfully")
print("Total faces:", len(embeddings))