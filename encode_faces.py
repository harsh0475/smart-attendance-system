import os
import pickle
import numpy as np
from deepface import DeepFace
import cv2

def encode_faces():
    dataset_path = "dataset"
    encodings = []
    names = []

    print("[INFO] Processing faces from dataset...")

    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)

        if not os.path.isdir(person_folder):
            continue

        print(f"[INFO] Encoding faces for: {person_name}")
        count = 0

        for img_file in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_file)

            try:
                # Get face embedding using DeepFace
                embedding = DeepFace.represent(
                    img_path=img_path,
                    model_name="VGG-Face",
                    enforce_detection=False
                )

                if embedding:
                    encodings.append(embedding[0]["embedding"])
                    names.append(person_name)
                    count += 1

            except Exception as e:
                print(f"[WARNING] Skipping {img_file}: {e}")
                continue

        print(f"[SUCCESS] Encoded {count} images for {person_name}")

    # Save encodings to file
    data = {"encodings": encodings, "names": names}
    with open("encodings.pkl", "wb") as f:
        pickle.dump(data, f)

    print(f"\n[SUCCESS] Total encodings saved: {len(encodings)}")
    print("[INFO] Saved to encodings.pkl")

if __name__ == "__main__":
    encode_faces()