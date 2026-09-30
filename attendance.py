import cv2
import os
import csv
from datetime import datetime

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

# Face detector
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# Get student names
students = sorted([
    name for name in os.listdir("dataset")
    if os.path.isdir(os.path.join("dataset", name))
])

# Attendance file
attendance_file = "attendance.csv"

if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Date", "Time"])

marked_students = set()

camera = cv2.VideoCapture(0)

print("Attendance system started...")
print("Press Q to stop.")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Camera not found")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        label, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if confidence < 70 and label < len(students):

            name = students[label]

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                name,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            if name not in marked_students:

                now = datetime.now()
                date = now.strftime("%Y-%m-%d")
                time = now.strftime("%H:%M:%S")

                with open(attendance_file, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([name, date, time])

                marked_students.add(name)

                print(f"Attendance marked for {name}")

        else:
            cv2.putText(
                frame,
                "Unknown",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

    cv2.imshow("Student Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("Attendance system stopped.")