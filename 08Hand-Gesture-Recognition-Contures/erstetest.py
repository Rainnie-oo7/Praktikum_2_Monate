import cv2
capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print("Error: Camera could not be opened.")
else:
    print("Camera opened successfully!")
