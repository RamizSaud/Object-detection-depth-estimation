import cv2
import numpy as np
import time
from matplotlib import pyplot as plt

# YOLO for object detection
from ultralytics import YOLO
import cvzone
import math

# Triangulation and calibration modules
import triangulation as tri
import calibration

# Load the YOLO model
model = YOLO("yolov8l.pt")

class_names = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant",
    "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Open both cameras
cap_right = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap_left = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Stereo vision setup parameters
frame_rate = 120    # Camera frame rate (maximum at 120 fps)
B = 9               # Distance between the cameras [cm]
f = 8               # Camera lens's focal length [mm]
alpha = 56.6        # Camera field of view in the horizontal plane [degrees]

# Main program loop with object detector and depth estimation using stereo vision
while cap_right.isOpened() and cap_left.isOpened():
    success_right, frame_right = cap_right.read()
    success_left, frame_left = cap_left.read()

    # YOLO detection on both frames
    results_right = model(frame_right)
    results_left = model(frame_left)

    # Calibration
    frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left)

    # If cannot catch any frame, break
    if not success_right or not success_left:
        break

    for r in results_right:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1,  y2 - y1
            cvzone.cornerRect(frame_right, (x1, y1, w, h))

            # Confidence
            conf = math.ceil(box.conf[0]*100)/100

            # Class Name
            cls = box.cls[0]

            cvzone.putTextRect(frame_right, f'{class_names[int(cls)]} {conf}', (max(0, x1), max(35, y1)), scale = 2, thickness = 2)

    for r in results_left:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1,  y2 - y1
            cvzone.cornerRect(frame_left, (x1, y1, w, h))

            # Confidence
            conf = math.ceil(box.conf[0]*100)/100

            # Class Name
            cls = box.cls[0]

            cvzone.putTextRect(frame_left, f'{class_names[int(cls)]} {conf}', (max(0, x1), max(35, y1)), scale = 2, thickness = 2)

    ################## CALCULATING DEPTH #########################################################
    if len(results_right.pred[0]) and len(results_left.pred[0]):
        # Assume the first detection is the object we are interested in
        # You can modify this part to select a specific object type
        detection_right = results_right.pred[0][0]
        detection_left = results_left.pred[0][0]

        x_right, y_right = detection_right[0], detection_right[1]
        x_left, y_left = detection_left[0], detection_left[1]

        center_point_right = (x_right + detection_right[2] / 2, y_right + detection_right[3] / 2)
        center_point_left = (x_left + detection_left[2] / 2, y_left + detection_left[3] / 2)

        # Calculate depth using triangulation
        depth = tri.find_depth(center_point_right, center_point_left, frame_right, frame_left, B, f, alpha)
        cv2.putText(frame_right, "Distance: " + str(round(depth, 1)) + " cm", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame_left, "Distance: " + str(round(depth, 1)) + " cm", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    else:
        cv2.putText(frame_right, "TRACKING LOST", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame_left, "TRACKING LOST", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frames
    cv2.imshow("frame right", frame_right)
    cv2.imshow("frame left", frame_left)

    # Hit "q" to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release and destroy all windows before termination
cap_right.release()
cap_left.release()
cv2.destroyAllWindows()
