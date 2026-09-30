import cv2
import os
import time

name = input("Enter student name: ")

folder = "dataset/" + name
os.makedirs(folder, exist_ok=True)

camera = cv2.VideoCapture(0)

count = 0

while count < 30:
    ret, frame = camera.read()

    if not ret:
        print("Camera not found")
        break

    cv2.imshow("Face Capture", frame)

    count += 1
    filename = f"{folder}/image_{count}.jpg"
    cv2.imwrite(filename, frame)

    print(f"Photo {count}/30 captured")

    time.sleep(0.3)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print(f"Captured {count} photos for {name}")