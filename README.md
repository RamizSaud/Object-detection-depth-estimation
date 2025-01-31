# Object Detection with Depth Estimation and Voice Assistance

## Overview
This project performs object detection with integrated depth estimation and voice feedback to assist visually impaired individuals. It utilizes YOLO for object detection, MiDaS and Stereo Vision for depth estimation, and a text-to-speech module for audio feedback.

## Video Demonstration
[![Watch the video](https://img.youtube.com/vi/mof2MxJGkNM/0.jpg)](https://www.youtube.com/watch?v=mof2MxJGkNM)

## Setup Instructions

### Prerequisites
- Python 3.8

### Installation Steps
1. Install Python 3.8 if not already installed.
2. Clone the repository:
   ```sh
   git clone https://github.com/RamizSaud/Object-detection-depth-estimation.git
   cd Object-detection-depth-estimation
   ```
3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Implementation Details

### Object Detection
- YOLO (You Only Look Once) is used for real-time object detection.
- Pre-trained YOLO models provide accurate object classification.

### Depth Estimation
- **MiDaS**: A deep learning-based monocular depth estimation model.
- **Stereo Vision**: Uses two camera images to estimate depth information.

### Voice Assistance
- Text-to-Speech (TTS) is integrated to provide real-time voice feedback on detected objects and their distances.

## Project Structure
```
|-- main.py              # YOLO implementation
|-- depth.py             # MiDaS implementation
|-- speak.py             # Text-to-Speech module
|-- model.py             # Entry point for the system integration
|-- requirements.txt     # List of dependencies
|-- README.md            # Project documentation
```

## Running the Project
To execute the system, run:
```sh
python model.py
```
This will initialize object detection, depth estimation, and voice feedback in real time.

## Contributors
- Muhammad Ramiz Saud

