import cv2
import numpy as np
import os

def event_triggered_sampling_and_saving(video_path, output_dir, flow_threshold=1.5):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        print("Could not read the video.")
        return

    # Convert the first frame to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_count = 0
    event_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Convert the current frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow between the previous and current frame
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Detect if the mean magnitude exceeds the threshold
        if np.mean(mag) > flow_threshold:
            event_frame_count += 1
            # Save the frame as an image
            image_path = os.path.join(output_dir, f"event_frame_{event_frame_count}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Saved {image_path}")

            # Update the previous frame to the current frame only if it's an event frame
            prev_gray = gray

    cap.release()
    print(f"Extracted and saved {event_frame_count} event-triggered frames.")

# Example usage
video_path = 'vidToImg/Input/trial.mp4'
output_dir = 'vidToImg/eventOutput'
flow_threshold = 1.5  # Set threshold for detecting significant motion

event_triggered_sampling_and_saving(video_path, output_dir, flow_threshold)
