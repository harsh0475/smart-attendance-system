# AI-Based Smart Attendance System using Face Recognition

An intelligent attendance management system that uses deep learning-based face recognition to automatically detect and mark attendance in real-time using a webcam.

---

## 🎯 Features

- Real-time face detection using OpenCV
- Deep learning face recognition using DeepFace (VGG-Face model)
- Automatic attendance marking with date and time
- Saves attendance records in CSV format
- Prevents duplicate attendance entries for the same day
- Supports multiple persons

---

## 🛠️ Tech Stack

- Python 3.10
- OpenCV - Camera and face detection
- DeepFace - Face recognition
- Pandas - Attendance CSV management
- NumPy - Numerical computations
- TensorFlow - Deep learning backend

---

## 📁 Project Structure
```
smart-attendance-system/
│
├── dataset/           ← Face images (not uploaded, add your own)
├── attendance/        ← Auto-generated attendance CSV files
├── add_person.py      ← Capture face images from webcam
├── encode_faces.py    ← Generate face encodings from dataset
├── app.py             ← Main real-time recognition app
├── requirements.txt   ← All dependencies
└── README.md          ← Project documentation
```

---

## ⚙️ Setup & Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/harsh0475/smart-attendance-system.git
cd smart-attendance-system
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Step 1: Add a Person to the System
```bash
python add_person.py
```
- Enter the person's name when prompted
- Look at the webcam
- 30 face images will be captured automatically

### Step 2: Generate Face Encodings
```bash
python encode_faces.py
```
- Processes all images from the dataset folder
- Saves face encodings to `encodings.pkl`

### Step 3: Run the Attendance System
```bash
python app.py
```
- Webcam opens automatically
- System detects and recognizes faces in real time
- Attendance is marked in `attendance/attendance_YYYY-MM-DD.csv`
- Press `q` to quit

---

## 📊 Attendance Output

Attendance is saved in CSV format inside the `attendance/` folder:

| Name | Date | Time | Status |
|---|---|---|---|
| Harshit | 2026-03-25 | 10:32:45 | Present |

---

## ⚠️ Notes

- Make sure your webcam is connected before running `app.py`
- Good lighting improves recognition accuracy
- Add multiple persons by running `add_person.py` multiple times
- Attendance is marked only once per person per day

---

## 📦 Requirements

- Python 3.10
- Webcam
- Windows 10/11

---

## 👤 Author

- **Name:** Harshit
- **GitHub:** [@harsh0475](https://github.com/harsh0475)