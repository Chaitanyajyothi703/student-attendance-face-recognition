import cv2
import os
import numpy as np
from PIL import Image

dataset_path = "dataset"

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []

label_id = 0

for student_name in os.listdir(dataset_path):
    student_folder = os.path.join(dataset_path, student_name)

    if not os.path.isdir(student_folder):
        continue

    print("Training:", student_name)

    for image_name in os.listdir(student_folder):
        image_path = os.path.join(student_folder, image_name)

        try:
            image = Image.open(image_path).convert("L")
            image_array = np.array(image, "uint8")

            faces.append(image_array)
            labels.append(label_id)

        except Exception as e:
            print("Error reading:", image_path)

    label_id += 1

if len(faces) == 0:
    print("No training images found!")
else:
    recognizer.train(faces, np.array(labels))
    recognizer.write("trainer.yml")

    print("Training completed successfully!")
    print("Model saved as trainer.yml")