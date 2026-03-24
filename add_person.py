import cv2
import os

def add_person():
    name = input("Enter the person's name: ").strip()
    
    # Create folder for this person
    person_folder = os.path.join("dataset", name)
    os.makedirs(person_folder, exist_ok=True)
    
    print(f"\n[INFO] Starting camera...")
    print(f"[INFO] Capturing 30 images for '{name}'")
    print(f"[INFO] Press 'q' to quit early\n")
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Could not open webcam!")
        return
    
    count = 0
    total = 30

    while count < total:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame!")
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            # Save face image
            face_img = frame[y:y+h, x:x+w]
            img_path = os.path.join(person_folder, f"{count}.jpg")
            cv2.imwrite(img_path, face_img)

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured: {count}/{total}",
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 0), 2)

        cv2.imshow("Adding Person - Press Q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[SUCCESS] Saved {count} images for '{name}'")
    print(f"[INFO] Images saved in: dataset/{name}/")

if __name__ == "__main__":
    add_person()