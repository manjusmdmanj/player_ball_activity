# player_ball_activity
# Player-Ball Activity Detection

This project involves detecting player-ball activities in a video using YOLOv8 and MediaPipe. The script processes video frames to detect players and balls, count leg touches, calculate ball rotation, and measure player velocity.

## Features

- **Player and Ball Detection**: Uses YOLOv8 for detecting players and balls in the video.
- **Pose Estimation**: Uses MediaPipe for detecting player's body parts and counting leg touches.
- **Ball Rotation Detection**: Analyzes ball rotation direction using optical flow.
- **Player Velocity Calculation**: Calculates the player's movement velocity between frames.

## Requirements

- Python 3.x
- OpenCV
- MediaPipe
- NumPy
- Ultralytics YOLO

Install the required packages using pip:

```bash
pip install opencv-python mediapipe numpy ultralytics
