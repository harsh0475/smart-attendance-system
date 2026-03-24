import cv2
import pickle
import numpy as np
import pandas as pd
from deepface import DeepFace
from datetime import datetime
import os

# ── Load encodings ──────────────────────────────────────────
def load_encodings(path="encodings.pkl"):
    if not os.path.exists(path):
        print("[ERROR] encodings.pkl not found! Run encode_faces.py first.")
        exit()
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"[INFO] Loaded {len(data['names'])} encodings.")
    return data["encodings"], data["names"]

# ── Compare faces ────────────────────────────────────────────
def find_match(face_embedding, known_encodings, known_names, threshold=0.4):
    min_dist = float("inf")
    match_name = "Unknown"

    for i, known_enc in enumerate(known_encodings):
        # Calculate cosine distance
        a = np.array(face_embedding)
        b = np.array(known_enc)
        cosine_dist = 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        if cosine_dist < min_dist:
            min_dist = cosine_dist
            if cosine_dist < threshold:
                match_name = known_names[i]

    return match_name, min_dist

# ── Mark Attendance ──────────────────────────────────────────
def mark_attendance(name):
    if name == "Unknown":
        return

    # Create attendance folder if not exists
    os.makedirs("attendance", exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    csv_path = f"attendance/attendance_{today}.csv"

    # Load existing attendance or create new
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=["Name", "Date", "Time", "Status"])

    # Check if already marked today
    if name in df["Name"].values:
        return

    # Add new entry
    now = datetime.now()
    new_row = {
        "Name": name,
        "Date": now.strftime("%Y-%m-%d"),
        "Time": now.strftime("%H:%M:%S"),
        "Status": "Present"
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"[ATTENDANCE] Marked Present: {name} at {now.strftime('%H:%M:%S')}")

# ── Main App ─────────────────────────────────────────────────
def main():
    known_encodings, known_names = load_encodings()

    print("[INFO] Starting webcam...")
    print("[INFO] Press 'q' to quit\n")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam!")
        return

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Process every 5th frame for performance
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Only recognize every 5th frame (saves CPU)
            if frame_count % 5 == 0:
                try:
                    face_img = frame[y:y+h, x:x+w]

                    # Get embedding for detected face
                    embedding = DeepFace.represent(
                        img_path=face_img,
                        model_name="VGG-Face",
                        enforce_detection=False
                    )

                    if embedding:
                        name, dist = find_match(
                            embedding[0]["embedding"],
                            known_encodings,
                            known_names
                        )

                        # Mark attendance
                        mark_attendance(name)

                        # Display name
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.putText(frame, f"{name} ({dist:.2f})",
                                   (x, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   0.8, color, 2)

                except Exception as e:
                    cv2.putText(frame, "Unknown",
                               (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               0.8, (0, 0, 255), 2)

        # Show date and time on screen
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, now_str, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Smart Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Attendance session ended.")
    print(f"[INFO] Check attendance/ folder for today's CSV file.")

if __name__ == "__main__":
    main()