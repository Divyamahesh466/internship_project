import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

class StateManager:
    def __init__(self):
        self.state = "OUTSIDE"

    def update(self, inside_zone):
        if inside_zone:
            self.state = "INSIDE"
        else:
            self.state = "OUTSIDE"
        return self.state


state_manager = StateManager()

person_id = 1
person_name = "Divya"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # ----------------------------
    # LEFT ZONE
    # ----------------------------
    zx1, zy1 = 0, 0
    zx2, zy2 = int(w * 0.4), h

    cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 255, 0), 3)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    inside_zone = False

    for (x, y, fw, fh) in faces:

        cx = x + fw // 2

        # zone check
        if cx < zx2:
            inside_zone = True

        state = state_manager.update(inside_zone)

        # ----------------------------
        # COLOR ONLY CHANGES
        # ----------------------------
        if state == "INSIDE":
            color = (255, 0, 0)   # BLUE
            label = "INSIDE"
        else:
            color = (0, 0, 255)   # RED
            label = "OUTSIDE"

        # ----------------------------
        # FACE BOX (ALWAYS SHOWN)
        # ----------------------------
        cv2.rectangle(frame, (x, y), (x+fw, y+fh), color, 2)

        # ----------------------------
        # NAME + ID ALWAYS SHOWN
        # ----------------------------
        cv2.putText(
            frame,
            f"{person_name} ID:{person_id}",
            (x, y - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # STATE SHOWN
        cv2.putText(
            frame,
            label,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imshow("ZONE SYSTEM", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()