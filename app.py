import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.title(" AI Helmet Detection System")
st.write("Upload image, video, or use webcam")

# Sidebar options
option = st.sidebar.selectbox(
    "Choose Input Type",
    ("Image", "Video", "Webcam")
)

# =========================
# IMAGE DETECTION
# =========================

if option == "Image":

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)
        frame = np.array(image)

        results = model(frame, conf=0.5)

        for result in results:

            for box in result.boxes:

                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                class_name = model.names[cls].lower()

                if class_name == "helmet":
                    color = (0,255,0)
                    label = f"Helmet {conf:.2f}"

                elif class_name == "nohelmet":
                    color = (0,0,255)
                    label = f"NoHelmet {conf:.2f}"

                else:
                    continue

                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                cv2.putText(frame,label,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

        st.image(frame, channels="BGR")

# =========================
# VIDEO DETECTION
# =========================

elif option == "Video":

    uploaded_file = st.file_uploader(
        "Upload Video",
        type=["mp4","avi","mov"]
    )

    if uploaded_file:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            results = model(frame, conf=0.5)

            for result in results:

                for box in result.boxes:

                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    class_name = model.names[cls].lower()

                    if class_name == "helmet":
                        color = (0,255,0)
                        label = f"Helmet {conf:.2f}"

                    elif class_name == "nohelmet":
                        color = (0,0,255)
                        label = f"NoHelmet {conf:.2f}"

                    else:
                        continue

                    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                    cv2.putText(frame,label,(x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

            stframe.image(frame, channels="BGR")

        cap.release()

# =========================
# WEBCAM DETECTION
# =========================

elif option == "Webcam":

    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:

        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame, conf=0.6)

        for result in results:

            for box in result.boxes:

                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                class_name = model.names[cls].lower()

                if class_name == "helmet":
                    color = (0,255,0)
                    label = f"Helmet {conf:.2f}"

                elif class_name == "nohelmet":
                    color = (0,0,255)
                    label = f"NoHelmet {conf:.2f}"

                else:
                    continue

                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                cv2.putText(frame,label,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

        FRAME_WINDOW.image(frame, channels="BGR")

    cap.release()