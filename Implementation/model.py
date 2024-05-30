import cv2
import torch
import math
from ultralytics import YOLO
import cvzone
import pyttsx3
import threading

# speaking mechanism
engine = pyttsx3.init()
def speak(obj, dis):
    def run():
        text = f"There is a {obj}, {dis} meters in front of you"
        engine.say(text)
        engine.runAndWait()
        
    threading.Thread(target=run).start()

# Load YOLO model for object detection
model_yolo = YOLO("yolov8l.pt")

# Load MiDaS model for depth estimation
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cuda')
midas.eval()

# Transformational Pipeline for MiDaS
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

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

ids = []

while True:
    success, img = cap.read()

    # Perform object detection using YOLO
    results = model_yolo.track(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1,  y2 - y1

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_batch = transform(img_rgb).to('cuda')

            # Make depth prediction
            with torch.no_grad():
                prediction = midas(img_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode='bicubic',
                    align_corners=False
                ).squeeze()

                depth_map = prediction.cpu().numpy()
                depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
                depth_map = depth_map.astype('uint8')

                # Crop the detected object for depth estimation
                depth_obj = depth_map[y1:y2, x1:x2]

                # Calculate mean pixel value of depth map
                max_depth = depth_obj.max()

                min_dist, max_dist = 0.5, 5.0
                distance = max_dist - (max_dist - min_dist) * (max_depth / 255.0)

                # Draw rectangle and text on image
                cvzone.cornerRect(img, (x1, y1, w, h))
                try:
                    text = f'id:{int(box.id)} {class_names[int(box.cls[0])]} {math.ceil(box.conf[0] * 100) / 100}, Distance: {distance:.2f}m'
                    if [int(box.id), class_names[int(box.cls[0])], round(distance, 1)] not in ids:
                        speak(class_names[int(box.cls[0])], round(distance, 1))
                        ids.append([int(box.id), class_names[int(box.cls[0])], round(distance, 1)])
                    cvzone.putTextRect(img, text, (max(0, x1), max(35, y1)), scale=1.5, thickness=2)
                except:
                    pass



    cv2.imshow("Image", img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
