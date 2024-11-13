from ultralytics import YOLO
import requests
from PIL import Image
from io import BytesIO
import os


# Download the YOLOv9 model if it doesn't exist
model_path = "yolov9m.pt"
# Load the pretrained YOLOv9 model
model = YOLO(model_path)


# Download the image from the URL
image_url = "https://di-uploads-pod25.dealerinspire.com/koenigseggflorida/uploads/2019/08/Koenigsegg_TheSquad_3200x2000-UPDATED.jpg"
response = requests.get(image_url)
img = Image.open(BytesIO(response.content))


# Set the confidence and IoU thresholds
confidence_threshold = 0.5
iou_threshold = 0.4


# Predict with the model using the set thresholds
results = model.predict(img, conf=confidence_threshold, iou=iou_threshold)

import cv2
from ultralytics import YOLO
import os

# Download the YOLOv9 model if it doesn't exist
model_path = "yolov9c.pt"
# Load the pretrained YOLOv9 model
model = YOLO(model_path)

# Initialize the webcam
# 0 is the default camera. If you have multiple cameras(ex: an external one on your laptop) you may need to change this number.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to the format expected by YOLO
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run YOLO model on the frame
    results = model.predict(frame_rgb)

    # Draw bounding boxes on the frame
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
        class_id = int(box.cls)
        class_name = results[0].names[class_id]
        confidence = box.conf.item()
        if confidence < .8:
            continue

        # Draw rectangle and label on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detections
    cv2.imshow("YOLO Webcam Detection", frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
