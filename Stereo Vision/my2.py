import cv2
import numpy as np
import cvzone
from ultralytics import YOLO

# Triangulation and calibration modules
import triangulation as tri
import calibration

# Load the YOLO model
model = YOLO("yolov8l.pt")

# Class names for detection
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
frame_rate = 120
B = 4
f = 8
alpha = 56.6

while cap_right.isOpened() and cap_left.isOpened():
    success_right, frame_right = cap_right.read()
    success_left, frame_left = cap_left.read()

    # Calibration
    frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left)

    # YOLO detection on both frames
    results_right = model(frame_right)
    results_left = model(frame_left)

    # If cannot catch any frame, break
    if not success_right or not success_left:
        break

    # Process detections in both frames
    # detections_right = [(box.xyxy, box.cls) for box in results_right.boxes]
    try:
        detections_right = [(box.xyxy[0], box.cls[0]) for box in [r.boxes for r in results_right]]
        # detections_left = [(box.xyxy, box.cls) for box in results_left.boxes]
        detections_left = [(box.xyxy[0], box.cls[0]) for box in [r.boxes for r in results_left]]

        # Loop through detections in the right frame
        for box_right, cls_right in detections_right:
            x1_right, y1_right, x2_right, y2_right = map(int, box_right)
            w_right , h_right = x2_right - x1_right, y2_right - y1_right

            # Find matching detection in left frame
            for box_left, cls_left in detections_left:
                if cls_right == cls_left:  # Ensure it's the same class
                    x1_left, y1_left, x2_left, y2_left = map(int, box_left)
                    w_left , h_left = x2_left - x1_left, y2_left - y1_left

                    # Calculate centers
                    center_right = ((x1_right + x2_right) // 2, (y1_right + y2_right) // 2)
                    center_left = ((x1_left + x2_left) // 2, (y1_left + y2_left) // 2)

                    # Calculate Depth
                    depth = tri.find_depth(center_right, center_left, frame_right, frame_left, B, f, alpha)
                    depth = abs(depth) / 100
                    # depth = depth / 2

                    # cv2.putText(frame_right, "Distance: " + str(round(depth, 1)) + " cm", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    cvzone.cornerRect(frame_right, (x1_right, y1_right, w_right, h_right))
                    cvzone.putTextRect(frame_right, f'{class_names[int(cls_right)]} {round(depth, 1)}m', (max(0, x1_right), max(35, y1_right)), scale = 2, thickness = 2)
                    # cv2.putText(frame_left, "Distance: " + str(round(depth, 1)) + " cm", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    cvzone.cornerRect(frame_left, (x1_left, y1_left, w_left, h_left))
                    cvzone.putTextRect(frame_left, f'{class_names[int(cls_left)]} {round(depth, 1)}m', (max(0, x1_left), max(35, y1_left)), scale = 2, thickness = 2)
    except:
        pass

    # Display the images
    cv2.imshow("Frame Right", frame_right)
    cv2.imshow("Frame Left", frame_left)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_right.release()
cap_left.release()
cv2.destroyAllWindows()
