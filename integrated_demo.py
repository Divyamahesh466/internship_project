import cv2

from config import *

from logger import (
    initialize_log,
    log_event
)

from state_manager import (
    StateManager
)

# -----------------------------
# Initialize
# -----------------------------

initialize_log()

state_manager = StateManager()

# -----------------------------
# Face Detector
# -----------------------------

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# -----------------------------
# Recognizer
# -----------------------------

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read(
    MODEL_PATH
)

# -----------------------------
# Labels
# -----------------------------

labels = {}

with open(
    LABELS_PATH,
    "r"
) as f:

    for line in f:

        idx, name = (
            line
            .strip()
            .split(",")
        )

        labels[int(idx)] = name

# -----------------------------
# Video
# -----------------------------

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():

    print("Video not found")
    exit()

# -----------------------------
# Main Loop
# -----------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # -------------------------
    # Draw Monitor Zone
    # -------------------------

    cv2.rectangle(
        frame,
        (ZONE_X1, ZONE_Y1),
        (ZONE_X2, ZONE_Y2),
        BOUNDARY_COLOR,
        3
    )

    cv2.putText(
        frame,
        "MONITOR ZONE",
        (ZONE_X1, ZONE_Y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        BOUNDARY_COLOR,
        2
    )

    # -------------------------
    # Face Detection
    # -------------------------

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        face_roi = gray[
            y:y+h,
            x:x+w
        ]

        # ---------------------
        # Recognition
        # ---------------------

        try:

            person_id, confidence = recognizer.predict(face_roi)


            if confidence < CONFIDENCE_THRESHOLD:

                person_name = labels[person_id]

                last_known_name = person_name

            else:

                person_name = last_known_name

        except:

            person_name = "Unknown"

        # ---------------------
        # Center Point
        # ---------------------

        cx = x + w // 2
        cy = y + h // 2

        inside_zone = (

            ZONE_X1 <= cx <= ZONE_X2

            and

            ZONE_Y1 <= cy <= ZONE_Y2

        )

        # ---------------------
        # State Machine
        # ---------------------

        current_state, event = (
            state_manager.update_state(
                person_name,
                inside_zone
            )
        )

        if (
            event is not None
            and
            person_name != "Unknown"
        ):

            log_event(
                person_name,
                event
            )

        # ---------------------
        # Draw Face Box
        # ---------------------

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            FACE_BOX_COLOR,
            2
        )

        cv2.putText(
            frame,
            f"{person_name}",
            (x, y-40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            TEXT_COLOR,
            2
        )

        cv2.putText(
            frame,
            f"State: {current_state}",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            TEXT_COLOR,
            2
        )

   
    cv2.imshow(
        "Face Tracking System",
        frame
    )

    if (
        cv2.waitKey(25)
        & 0xFF
        == ord("q")
    ):
        break

cap.release()

cv2.destroyAllWindows()