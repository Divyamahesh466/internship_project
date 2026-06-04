import cv2
import os
import numpy as np
from PIL import Image

# ----------------------------
# Paths
# ----------------------------
DATASET_PATH = "dataset"
MODEL_PATH = "models"

os.makedirs(MODEL_PATH, exist_ok=True)

# ----------------------------
# Face Detector
# ----------------------------
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# ----------------------------
# LBPH Recognizer
# ----------------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()

face_samples = []
ids = []

label_names = {}
current_id = 0

# ----------------------------
# Read Dataset
# ----------------------------
for person_name in os.listdir(DATASET_PATH):

    person_folder = os.path.join(
        DATASET_PATH,
        person_name
    )

    if not os.path.isdir(person_folder):
        continue

    print(f"[INFO] Processing {person_name}")

    label_names[current_id] = person_name

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(
            person_folder,
            image_name
        )

        try:
            img = Image.open(
                image_path
            ).convert("L")

            img_numpy = np.array(
                img,
                "uint8"
            )

            faces = detector.detectMultiScale(
                img_numpy,
                scaleFactor=1.1,
                minNeighbors=5
            )

            for (x, y, w, h) in faces:

                face_samples.append(
                    img_numpy[y:y+h, x:x+w]
                )

                ids.append(current_id)

        except Exception as e:
            print(
                f"Error reading {image_path}: {e}"
            )

    current_id += 1

# ----------------------------
# Train Model
# ----------------------------
print(
    f"\nTraining using "
    f"{len(face_samples)} face samples..."
)

recognizer.train(
    face_samples,
    np.array(ids)
)

# ----------------------------
# Save Model
# ----------------------------
trainer_file = os.path.join(
    MODEL_PATH,
    "trainer.yml"
)

recognizer.save(trainer_file)

# ----------------------------
# Save Labels
# ----------------------------
labels_file = os.path.join(
    MODEL_PATH,
    "labels.txt"
)

with open(labels_file, "w") as f:

    for id_, name in label_names.items():

        f.write(
            f"{id_},{name}\n"
        )

print("\nTraining Completed")
print(
    f"Model saved -> {trainer_file}"
)
print(
    f"Labels saved -> {labels_file}"
)