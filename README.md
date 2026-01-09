# ROSbot 2 Lane Following System

An end-to-end autonomous lane-following system deployed on ROSbot 2, integrating deep learning-based perception, PID control, and real-time visualization using ROS 2.

## Overview

This project implements a complete lane-following pipeline for the ROSbot 2 mobile robot platform. The system uses a U-Net segmentation model trained on 26,000 images to detect lane boundaries in real-time, combined with a PID controller for smooth steering control.

## Features

- **Deep Learning Perception:** U-Net based lane boundary segmentation
- **Real-time Processing:** Image preprocessing pipeline with OpenCV
- **Control System:** Custom PID controller for stable steering
- **ROS 2 Integration:** Full ROS 2 package with topic-based communication
- **Visualization:** RViz-based debugging and behavior monitoring

## Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | ROS 2 |
| Language | Python |
| Segmentation Model | U-Net |
| Computer Vision | OpenCV |
| Control | PID Controller |
| Visualization | RViz |
| Platform | ROSbot 2 |

## Project Structure

```
├── images/                    # Training images
├── masks/                     # Segmentation masks
├── annotations/               # Image annotations
├── predictions/               # Model predictions
├── epoch_outputs/             # Training epoch results
├── rosbot_lane_follower/      # Main ROS 2 package
├── ros_ws/                    # ROS 2 workspace
├── ros_ws2/                   # Additional workspace
├── ros_ws3/                   # Additional workspace
├── train_unet/                # U-Net training scripts
├── topic_data/                # ROS topic data
├── topicdata_ex/              # Extended topic data
├── video_from_images/         # Video generation scripts
├── visualize_predictions/     # Prediction visualization
├── input_video/               # Input video files
├── lane_detected_video/       # Output with lane detection
├── predict_video/             # Video prediction scripts
├── lane_unet.pth              # Trained U-Net model weights
├── commands                   # Useful commands reference
└── extract_odometry_from_text # Odometry extraction utilities
```

## Model Details

- **Architecture:** U-Net (Encoder-Decoder with skip connections)
- **Training Dataset:** 26,000 custom annotated images
- **Input:** RGB camera feed from ROSbot 2
- **Output:** Binary lane mask for boundary detection

## Pipeline

```
Camera Feed → Preprocessing → U-Net Segmentation → Lane Detection → PID Control → Motor Commands
                                                          ↓
                                                   RViz Visualization
```

## Installation

```bash
# Clone the repository
git clone https://github.com/adomynx/rosbot2-lane-follower.git
cd rosbot2-lane-follower

# Build ROS 2 workspace
cd ros_ws
colcon build
source install/setup.bash
```

## Usage

```bash
# Launch the lane following system
ros2 launch rosbot_lane_follower lane_follower.launch.py

# Visualize in RViz
ros2 run rviz2 rviz2
```

## Results

- Robust lane detection in varying lighting conditions
- Smooth and stable steering using PID control
- Real-time performance on ROSbot 2 hardware
- Validated through simulation demos and live testing

## Demo

[Add demo GIF or video link here]

## Author

**Kartik Wakekar**
- LinkedIn: [Kartik-Wakekar](https://linkedin.com/in/Kartik-Wakekar)
- GitHub: [adomynx](https://github.com/adomynx)

