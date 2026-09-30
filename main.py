import cv2

print("OpenCV loaded")
print("Version:", cv2.__version__)

recognizer = cv2.face.LBPHFaceRecognizer_create()

print("Face recognizer is working!")