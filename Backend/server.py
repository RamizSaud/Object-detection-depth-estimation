import asyncio
import cv2
import torch
import math
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from ultralytics import YOLO
import cvzone
from av import VideoFrame

# Load YOLO model for object detection
model_yolo = YOLO("yolov8l.pt")

# Load MiDaS model for depth estimation
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cuda')
midas.eval()

# Transformational Pipeline for MiDaS
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

class VideoProcessorTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track):
        super().__init__()
        self.track = track

        # Define class names
        self.class_names = [
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

    async def recv(self):
        frame = await self.track.recv()
        print("Received frame")

        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")

        # Perform object detection using YOLO
        results = model_yolo(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                w, h = x2 - x1, y2 - y1

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

                    depth_obj = depth_map[y1:y2, x1:x2]
                    max_depth = depth_obj.max()

                    min_dist, max_dist = 0.5, 5.0
                    distance = max_dist - (max_dist - min_dist) * (max_depth / 255.0)

                    # Get class name using class index
                    class_name = self.class_names[int(box.cls[0])]

                    cvzone.cornerRect(img, (x1, y1, w, h))
                    text = f'{class_name} {math.ceil(box.conf[0] * 100) / 100}, Distance: {distance:.2f}m'
                    cvzone.putTextRect(img, text, (max(0, x1), max(35, y1)), scale=1, thickness=2)

        cv2.imshow("Processed Image", img)
        cv2.waitKey(1)

        # height, width, _ = img.shape
        processed_frame = VideoFrame.from_ndarray(img, format="bgr24")
        processed_frame.pts = frame.pts
        processed_frame.time_base = frame.time_base

        return processed_frame

        # return frame  # You might want to modify this part based on your application's needs


async def index(request):
    return web.Response(content_type="text/html", text="WebRTC Server Running")

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % id(pc)
    print(f"{pc_id} Created for offer")

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        print(f"{pc_id} ICE connection state is {pc.iceConnectionState}")
        if pc.iceConnectionState == "failed":
            await pc.close()

    @pc.on("track")
    def on_track(track):
        print(f"{pc_id} Track {track.kind} received")
        if track.kind == "video":
            local_video = VideoProcessorTrack(track)
            pc.addTrack(local_video)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })

async def main():
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 5000)
    await site.start()

    print("Server started at http://0.0.0.0:5000")
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
