import cv2
from ultralytics import YOLO
import os
import csv
from datetime import datetime
import time

result_folder = "Result"
violation_folder = os.path.join(result_folder, "Violation")

os.makedirs(result_folder, exist_ok=True)
os.makedirs(violation_folder, exist_ok=True)

csv_file = os.path.join(result_folder, "results.csv")

# Create CSV
if not os.path.exists(csv_file):
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp","Helmets","Violations","Compliance","Status"])

model = YOLO("best.pt")

print("Model loaded successfully")

cap = cv2.VideoCapture(0)

print("Press ESC to exit")

last_logged_time = 0
cooldown_seconds = 5

while True:

    ret, frame = cap.read()

    if not ret:
        break

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
                color=(0,255,0)
                helmet_count+=1

            elif class_name == "nohelmet":
                color=(0,0,255)
                violation_count+=1

            else:
                continue

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

    total = helmet_count + violation_count

    compliance = (helmet_count/total)*100 if total>0 else 0

    status = "Violation Detected" if violation_count>0 else "Safe"

    current_time = time.time()

    if violation_count>0 and (current_time-last_logged_time)>cooldown_seconds:

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(csv_file,"a",newline="") as file:

            writer = csv.writer(file)

            writer.writerow([
                timestamp,
                helmet_count,
                violation_count,
                round(compliance,2),
                status
            ])

        filename=datetime.now().strftime("%Y%m%d_%H%M%S")+".jpg"

        path=os.path.join(violation_folder,filename)

        cv2.imwrite(path,frame)

        last_logged_time=current_time

    cv2.imshow("Helmet Detection Test",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()
