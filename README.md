# AI-Based Smart Attendance System using Face Recognition

An intelligent attendance management system that uses deep learning-based face
recognition to automatically detect and mark attendance in real time using a webcam.
Instead of manually calling names or signing registers, this system identifies a
person's face through the camera and instantly records their attendance in a CSV file
with the exact date and time.

---

## What This Project Does

- Opens your webcam and detects faces in real time
- Recognizes the person using a deep learning face recognition model (VGG-Face)
- Automatically marks attendance in a CSV file with name, date, and time
- Prevents duplicate entries — marks each person only once per day
- Shows a green box with the person's name if recognized
- Shows a red box with "Unknown" if the person is not in the system
- Supports multiple persons

---

## How It Works (Simple Overview)
```
Step 1 → Collect face images of each person using the webcam
Step 2 → Convert face images into numerical data (face encodings)
Step 3 → Run the system → it recognizes faces and marks attendance automatically
```

---

## Prerequisites

Before you begin, make sure you have the following installed on your system:

- **Python 3.10** — Download from https://www.python.org/downloads/
  - During installation, check "Add Python to PATH"
  - Python 3.11 and above may cause compatibility issues
- **Git** — Download from https://git-scm.com/downloads
- **A working webcam** — Built-in or external USB webcam
- **Stable internet connection** — Required for first run (downloads the VGG-Face
  model, approximately 500MB)
- **Windows 10 or 11** — This project is tested on Windows

---

## Project Structure
```
smart-attendance-system/
│
├── add_person.py       ← Run this to add a new person to the system
├── encode_faces.py     ← Run this to process face images into encodings
├── app.py              ← Run this to start the attendance system
├── requirements.txt    ← All required Python libraries
├── README.md           ← You are reading this
│
├── dataset/            ← Created automatically when you add a person
│   └── PersonName/     ← 30 face images stored here per person
│
├── attendance/         ← Created automatically when attendance is marked
│   └── attendance_YYYY-MM-DD.csv  ← Daily attendance file
│
└── encodings.pkl       ← Created automatically after running encode_faces.py
```

> Note: dataset/, attendance/, and encodings.pkl are not included in this
> repository. They are generated automatically when you run the scripts.

---

## Setup and Installation

### Step 1 — Clone the Repository
Open Command Prompt and run:
```bash
git clone https://github.com/harsh0475/smart-attendance-system.git
cd smart-attendance-system
```

### Step 2 — Create a Virtual Environment
```bash
python -m venv venv
```

### Step 3 — Activate the Virtual Environment
```bash
venv\Scripts\activate
```
You will see `(venv)` appear at the start of your terminal line. This means
the virtual environment is active.

### Step 4 — Install All Dependencies
```bash
pip install -r requirements.txt
```
This will install all required libraries including OpenCV, DeepFace, TensorFlow,
Pandas, and NumPy. This may take 5 to 10 minutes depending on your internet speed.

---

## How to Use

### Step 1 — Add a Person to the System
```bash
python add_person.py
```
**What happens:**
- You will be asked to enter the person's name
- The webcam will open automatically
- Look directly at the camera
- The system will automatically capture 30 face images
- Images are saved in dataset/PersonName/ folder
- The webcam closes automatically after 30 images are captured

**Run this step once for each person you want to add.**

---

### Step 2 — Generate Face Encodings
```bash
python encode_faces.py
```
**What happens:**
- The system reads all face images from the dataset folder
- Each face is processed through the VGG-Face deep learning model
- Face encodings (numerical representations) are generated
- All encodings are saved to encodings.pkl
- On first run, the VGG-Face model (~500MB) will be downloaded automatically

**You must run this step every time you add a new person.**

---

### Step 3 — Run the Attendance System
```bash
python app.py
```
**What happens:**
- The webcam opens automatically
- The system detects and recognizes faces in real time
- If your face is recognized:
  - A green box appears around your face
  - Your name is displayed on screen
  - Your attendance is marked in the CSV file
- If the face is not recognized:
  - A red box appears
  - "Unknown" is displayed on screen
- Press `q` on your keyboard to stop the system

---

## Output

### On Screen
```
Green box + Name    → Person recognized, attendance marked
Red box + Unknown   → Person not in the system
```

### In Terminal
```
[INFO] Loaded 30 encodings.
[INFO] Starting webcam...
[INFO] Press 'q' to quit
[ATTENDANCE] Marked Present: Harshit at 10:32:45
[INFO] Attendance session ended.
[INFO] Check attendance/ folder for today's CSV file.
```

### Attendance CSV File
A file named `attendance_YYYY-MM-DD.csv` is created inside the `attendance/`
folder. Example:

| Name    | Date       | Time     | Status  |
|---------|------------|----------|---------|
| Harshit | 2026-03-25 | 10:32:45 | Present |

---

## Adding Multiple Persons

To add more than one person, run `add_person.py` multiple times:
```bash
python add_person.py   ← Enter "Person1" when asked
python add_person.py   ← Enter "Person2" when asked
python add_person.py   ← Enter "Person3" when asked
python encode_faces.py ← Run this once after adding all persons
python app.py          ← System now recognizes all persons
```

---

## Troubleshooting

**Webcam not opening:**
- Make sure your webcam is connected and not being used by another application
- Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in app.py

**Face not being recognized:**
- Make sure you are in good lighting
- Make sure you ran encode_faces.py after adding your face
- Try re-adding your face using add_person.py with better lighting

**encodings.pkl not found error:**
- You forgot to run encode_faces.py
- Run `python encode_faces.py` before running app.py

**pip install fails:**
- Make sure your virtual environment is activated (you should see `(venv)`)
- Make sure you are using Python 3.10

---

## Tech Stack

| Library | Purpose |
|---|---|
| Python 3.10 | Core programming language |
| OpenCV | Webcam access and face detection |
| DeepFace | Deep learning face recognition |
| VGG-Face | Pre-trained face recognition model |
| TensorFlow | Deep learning backend |
| NumPy | Cosine distance calculation |
| Pandas | Attendance CSV management |

---

##  

- **Name:** Harshit Kumar Singh
- **GitHub:** [@harsh0475](https://github.com/harsh0475)