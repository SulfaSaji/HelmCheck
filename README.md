# 🪖 AI Helmet Detection System (YOLOv8)

An AI-powered **Helmet Violation Detection System** built using **YOLOv8 and Computer Vision**.

The system detects whether motorcycle riders are wearing helmets or not from **images, videos, and live webcam streams**.

Detection results are visualized with bounding boxes:

🟢 Helmet detected → Green bounding box  
🔴 No Helmet detected → Red bounding box  

The project also includes a **Streamlit web application dashboard** for real-time interaction and monitoring.

---

# 📌 Overview

Road safety is a major concern worldwide. Many motorcycle accidents become fatal due to riders not wearing helmets.

This project uses **Deep Learning and Computer Vision** to automatically detect helmet usage and identify safety violations.

The system can:

- Detect helmets in images
- Detect helmets in videos
- Perform **real-time detection using webcam**
- Highlight violations visually
- Automatically **save violation evidence**
- Record violations in a **CSV database**
- Provide an interactive **Streamlit web interface**

This system can assist **traffic monitoring systems and smart surveillance solutions**.

---

# 📂 Dataset

The model was trained using multiple helmet detection datasets merged together to improve performance and accuracy.

The dataset contains **two classes**:

| Class | Description |
|------|-------------|
| Helmet | Rider wearing helmet |
| NoHelmet | Rider without helmet |

Dataset format follows **YOLO annotation format**.

Example dataset structure:

```
dataset
 ├── train
 │   ├── images
 │   ├── labels
 │
 ├── valid
 │   ├── images
 │   ├── labels
 │
 └── test
     ├── images
     ├── labels
```

For demonstration purposes, sample images and videos are provided in:

```
demo test dataset
 ├── images
 └── videos
```

---

# 🛠 Tools & Technologies

The project was developed using:

- Python
- YOLOv8 (Ultralytics)
- PyTorch
- OpenCV
- NumPy
- Pandas
- Streamlit

---

# ⚙️ Project Workflow

### 1️⃣ Data Preparation

- Collect helmet detection datasets
- Merge datasets
- Convert annotations to YOLO format

### 2️⃣ Model Training

- Train YOLOv8 model on helmet dataset
- Detect two classes: `Helmet` and `NoHelmet`

### 3️⃣ Model Export

After training, the model generates:

```
best.pt
```

This file contains the **trained YOLOv8 weights** used for detection.

### 4️⃣ Detection System

The trained model is integrated into scripts that support:

- Image detection
- Video detection
- Live webcam detection

### 5️⃣ Web Application

A **Streamlit dashboard** was built to allow users to upload media and perform detection interactively.

---

# 📊 Dashboard Features

The Streamlit interface provides:

✔ Image upload detection  
✔ Video upload detection  
✔ Live webcam detection  
✔ Color-coded detection boxes  
✔ Easy-to-use web interface  

Detection visualization:

| Detection | Color |
|------|------|
| Helmet | 🟢 Green |
| No Helmet | 🔴 Red |

---

# 📈 Violation Logging & Results

The system automatically logs helmet violations.

Whenever **NoHelmet** is detected:

- The violation frame is **saved as an image**
- Violation details are **recorded in a CSV file**

This allows tracking and analysis of safety violations.

---

# 📁 Result Folder Structure

```
Result
│
├── Violation
│   ├── violation_1.jpg
│   ├── violation_2.jpg
│   ├── violation_3.jpg
│
└── results.csv
```

---

# 📄 Violation CSV Log

All violations are recorded inside:

```
Result/results.csv
```

Example structure:

| Timestamp | Helmets | Violations | Compliance | Status |
|----------|--------|-----------|------------|-------|
| 2026-03-08 15:20 | 3 | 1 | 75% | Violation Detected |

This enables **violation tracking and safety analysis**.

---

# 🚀 How to Run the Project

## 1️⃣ Clone the Repository

```
git clone https://github.com/SulfaSaji/AI-Helmet-Detection-System.git
cd AI-Helmet-Detection-System
```

---

## 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

## 3️⃣ Run Detection Script

```
python detect.py
```

This runs **real-time webcam detection** using OpenCV.

---

## 4️⃣ Launch the Web Application

```
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

# 📊 Project Presentation

The repository also includes the project presentation slides.

```
AI-Helmet-Detection-System.pptx
```

The presentation explains:

- Problem statement
- Dataset preparation
- Model training
- Detection pipeline
- Streamlit dashboard
- Violation logging system
- Future improvements

---

# 📁 Project Structure

```
AI-Helmet-Detection-System
│
├── app.py
├── detect.py
├── best.pt
├── requirements.txt
├── README.md
├── AI-Helmet-Detection-System.pptx
│
├── demo test dataset
│   ├── images
│   └── videos
│
└── Result
    ├── Violation
    └── results.csv
```

---

# 🔮 Future Improvements

Possible enhancements include:

- License plate recognition
- Integration with CCTV traffic cameras
- Cloud-based monitoring dashboard
- Automated violation reporting
- Real-time traffic analytics

---

# 👩‍💻 Author

**Sulfa Saji**

