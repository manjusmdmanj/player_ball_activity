import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# Initialize YOLOv8 model for ball and player detection
model = YOLO(r"D:\flick_it_demo\yolov8n.pt")  # Use the tiny model for faster inference

# MediaPipe Pose Model for detecting body parts
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize video input
cap = cv2.VideoCapture(r"D:\flick_it_demo\Toe Taps.mp4")  # Replace with your video path

# Define the codec and create a VideoWriter object to save the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('annotated_output_1.avi', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))  # Adjust frame rate and resolution

# Variables to count leg touches
right_leg_touch_count = 0
left_leg_touch_count = 0
prev_player_position = None
prev_gray = None

# Read the first frame
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Define a function to overlay the results on the video
def overlay_results(frame, right_leg_count, left_leg_count, rotation, velocity):
    # Display touch counts
    cv2.putText(frame, f'Right Leg Touches: {right_leg_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Left Leg Touches: {left_leg_count}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display ball rotation direction
    cv2.putText(frame, f'Ball Rotation: {rotation}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display player velocity
    cv2.putText(frame, f'Player Velocity: {velocity:.2f} px/frame', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Calculate the player velocity based on position change
def calculate_velocity(current_position, prev_position):
    if prev_position is None:
        return 0
    distance = np.linalg.norm(np.array(current_position) - np.array(prev_position))
    return distance

# Check if the ball is near the player's leg
def is_touching(ball_position, leg_position):
    distance = np.linalg.norm(np.array(ball_position) - np.array(leg_position))
    return distance < 50  # Threshold for considering a touch

# Main loop to process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection for player and ball detection
    results = model(frame)
    
    # Check if results is a list and process accordingly
    if isinstance(results, list):
        detections = results[0].boxes  # Get the first result's boxes

        ball_position = None
        player_position = None

        # Extract ball and player positions from YOLO detections
        for box in detections:
            xyxy = box.xyxy.numpy()  # Convert to numpy array
            label = int(box.cls.numpy())  # Class label
            
            # Handle xyxy to extract bounding box coordinates
            xyxy = np.array(xyxy)
            if xyxy.ndim > 1:
                for bbox in xyxy:
                    x1, y1, x2, y2 = bbox
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    # Draw bounding boxes on the detected objects
                    if label == 0:  # Class 0: Person
                        player_position = (int(center_x), int(center_y))
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, 'Player', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    elif label == 32:  # Class 32: Sports Ball
                        ball_position = (int(center_x), int(center_y))
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        cv2.putText(frame, 'Ball', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Pose estimation for the player's body parts
    pose_results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if pose_results.pose_landmarks and player_position and ball_position:
        landmarks = pose_results.pose_landmarks.landmark

        # Get leg positions
        left_leg_position = (int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * frame.shape[1]),
                             int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * frame.shape[0]))
        right_leg_position = (int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * frame.shape[1]),
                              int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * frame.shape[0]))

        # Draw leg landmarks
        cv2.circle(frame, left_leg_position, 5, (255, 0, 0), -1)
        cv2.circle(frame, right_leg_position, 5, (255, 0, 0), -1)
        
        # Check if the ball is touching the right or left leg
        if is_touching(ball_position, right_leg_position):
            right_leg_touch_count += 1
        if is_touching(ball_position, left_leg_position):
            left_leg_touch_count += 1

    # Calculate ball rotation using optical flow
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    if ball_position:
        flow_x, flow_y = flow[ball_position[1], ball_position[0]]
        if flow_x > 0:
            ball_rotation = "Forward"
        else:
            ball_rotation = "Backward"
    else:
        ball_rotation = "Unknown"

    prev_gray = gray  # Update the previous frame for optical flow

    # Calculate player velocity
    velocity = 0
    if player_position:
        velocity = calculate_velocity(player_position, prev_player_position)
    prev_player_position = player_position

    # Overlay results on the video
    overlay_results(frame, right_leg_touch_count, left_leg_touch_count, ball_rotation, velocity)

    # Write the annotated frame to the video file
    out.write(frame)

    # Display the frame with annotations
    cv2.imshow("Annotated Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()  # Release the video writer object
cv2.destroyAllWindows()
