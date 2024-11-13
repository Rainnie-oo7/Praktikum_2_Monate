
import numpy as np
import cv2

# Hold the background frame for background subtraction.
background = None
# Hold the hand's PennFudanPed so all its details are in one place.
hand = None
# Variables to count how many frames have passed and to set the size of the window.
frames_elapsed = 0
FRAME_HEIGHT = 200
FRAME_WIDTH = 300
# Humans come in a ton of beautiful shades and colors.
# Try editing these if your program has trouble recognizing your skin tone.
CALIBRATION_TIME = 30
BG_WEIGHT = 0.5
OBJ_THRESHOLD = 18

# capture = cv2.VideoCapture(1)
# if not capture.isOpened():
#     print("Error: Camera could not be opened.")
# else:
#     print("Camera opened successfully!")

### Add a check after capture.read() to ensure the frame is captured correctly:
#
# ret, frame = capture.read()
# if not ret:
#     print("Error: Could not read frame from camera.")

# ### Main function: Get input from camera and call functions to understand it

capture = cv2.VideoCapture(0)  # Ensure the correct camera index
FRAME_WIDTH, FRAME_HEIGHT = 640, 480

if not capture.isOpened():
    print("Error: Camera could not be opened.")
    exit()

while True:
    ret, frame = capture.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    # frame = cv2.flip(frame, 1)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()



