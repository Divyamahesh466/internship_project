import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis

DATASET_PATH = "Data Set"
OUTPUT_FILE = "embeddings.pkl"

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=0, det_size=(640, 640))

known_embeddings = []
known_names = []

for person_name in os.listdir(DATASET_PATH):

    person_folder = os.path.join(DATASET_PATH, person_name)

    if not os.path.isdir(person_folder):
        continue

    print(f"[INFO] Processing {person_name}")

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(person_folder, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        faces = app.get(image)

        if len(faces) == 0:
            print(f"[WARNING] No face found: {image_path}")
            continue

        embedding = faces[0].embedding

        known_embeddings.append(embedding)
        known_names.append(person_name)

data = {
    "embeddings": np.array(known_embeddings),
    "names": known_names
}

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(data, f)

print("Training Completed")
print(f"Saved embeddings -> {OUTPUT_FILE}")