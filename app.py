import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import csv
from datetime import datetime
import pandas as pd

# =========================
# CREATE RESULT FOLDERS
# =========================

result_folder = "Result"
violation_folder = os.path.join(result_folder, "Violation")

os.makedirs(result_folder, exist_ok=True)
os.makedirs(violation_folder, exist_ok=True)

csv_file = os.path.join(result_folder, "results.csv")

# Create CSV if not exists
if not os.path.exists(csv_file):
    with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp","Helmets","Violations","Compliance","Status"])

# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.title("AI Helmet Detection System")

# =========================
# SIDEBAR
# =========================

st.sidebar.title("Menu")

menu = st.sidebar.radio(
    "Navigation",
    ("Detection","Results Dashboard")
)

if menu == "Detection":

    option = st.sidebar.selectbox(
        "Choose Input Type",
        ("Image","Video","Webcam")
    )

else:

    result_option = st.sidebar.selectbox(
        "Results Dashboard",
        ("CSV Logs","Violation Images")
    )

# =========================
# DETECTION FUNCTION
# =========================

def process_frame(frame):

    helmet_count = 0
    violation_count = 0

    results = model(frame, conf=0.5)

    for result in results:

        for box in result.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            x1,y1,x2,y2 = map(int, box.xyxy[0])

            class_name = model.names[cls].lower()

            if class_name == "helmet":
                color = (0,255,0)
                label = f"Helmet {conf:.2f}"
                helmet_count += 1

            elif class_name == "nohelmet":
                color = (0,0,255)
                label = f"NoHelmet {conf:.2f}"
                violation_count += 1

            else:
                continue

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

            cv2.putText(frame,label,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    total = helmet_count + violation_count

    compliance = (helmet_count/total)*100 if total>0 else 0

    status = "Violation Detected" if violation_count>0 else "Safe"

    return frame,helmet_count,violation_count,compliance,status

# =========================
# IMAGE DETECTION
# =========================

if menu=="Detection" and option=="Image":

    uploaded_file = st.file_uploader("Upload Image",type=["jpg","jpeg","png"])

    if uploaded_file:

        image = Image.open(uploaded_file)
        frame = np.array(image)

        frame,helmet_count,violation_count,compliance,status = process_frame(frame)

        st.image(frame,channels="BGR")

        st.subheader("Detection Statistics")

        st.write("Helmets :",helmet_count)
        st.write("Violations :",violation_count)
        st.write("Compliance :",round(compliance,2),"%")
        st.write("Status :",status)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(csv_file,"a",newline="",encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([timestamp,helmet_count,violation_count,round(compliance,2),status])

        if violation_count>0:

            filename=datetime.now().strftime("%Y%m%d_%H%M%S")+".jpg"
            path=os.path.join(violation_folder,filename)

            cv2.imwrite(path,frame)

# =========================
# VIDEO DETECTION
# =========================

elif menu=="Detection" and option=="Video":

    uploaded_file = st.file_uploader("Upload Video",type=["mp4","avi","mov"])

    if uploaded_file:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        while cap.isOpened():

            ret,frame = cap.read()

            if not ret:
                break

            frame,helmet_count,violation_count,compliance,status = process_frame(frame)

            stframe.image(frame,channels="BGR")

        cap.release()

# =========================
# WEBCAM DETECTION
# =========================

elif menu=="Detection" and option=="Webcam":

    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:

        ret,frame = cap.read()

        if not ret:
            break

        frame,helmet_count,violation_count,compliance,status = process_frame(frame)

        FRAME_WINDOW.image(frame,channels="BGR")

        if violation_count>0:

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(csv_file,"a",newline="",encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([timestamp,helmet_count,violation_count,round(compliance,2),status])

            filename=datetime.now().strftime("%Y%m%d_%H%M%S")+".jpg"
            path=os.path.join(violation_folder,filename)

            cv2.imwrite(path,frame)

    cap.release()

# =========================
# CSV LOGS
# =========================

elif menu=="Results Dashboard" and result_option=="CSV Logs":

    st.header("CSV Violation Logs")

    if os.path.exists(csv_file):

        df = pd.read_csv(csv_file)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.dropna(how="all")

        st.dataframe(df)

        with open(csv_file,"rb") as file:

            st.download_button(
                "Download CSV",
                file,
                file_name="results.csv"
            )

# =========================
# IMAGE VIEWER
# =========================

elif menu=="Results Dashboard" and result_option=="Violation Images":

    st.header("Violation Image Folder")

    images = os.listdir(violation_folder)

    if len(images)==0:

        st.info("No images saved")

    else:

        selected_image = st.selectbox("Select Image File",images)

        path = os.path.join(violation_folder,selected_image)

        st.image(path,width=450,caption=selected_image)
