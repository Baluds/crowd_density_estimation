import cv2
import numpy as np
import os

def event_triggered_sampling_and_saving(
    video_path,
    output_dir,
    normal_out_dir,
    flow_threshold=1.5,
    min_frames_between_events=30,
    resize_width=None
):
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(normal_out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame of the video.")
        cap.release()
        return

    if resize_width is not None:
        aspect_ratio = prev_frame.shape[1] / prev_frame.shape[0]
        resize_height = int(resize_width / aspect_ratio)
        prev_frame = cv2.resize(prev_frame, (resize_width, resize_height))

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_count = 1
    frame_count_normal = 1
    event_frame_count = 0
    frames_since_last_event = min_frames_between_events

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if resize_width is not None:
            frame = cv2.resize(frame, (resize_width, resize_height))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, 
            None, 
            pyr_scale=0.5, 
            levels=3, 
            winsize=15, 
            iterations=3, 
            poly_n=5, 
            poly_sigma=1.2, 
            flags=0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        if np.mean(mag) > flow_threshold and frames_since_last_event >= min_frames_between_events:
            event_frame_count += 1
            frame_count_normal += 1
            image_path = os.path.join(output_dir, f"event_frame_{frame_count_normal:04d}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Saved: {image_path}")

            prev_gray = gray.copy()
            frames_since_last_event = 0
        else:
            frame_count_normal += 1
            image_path = os.path.join(normal_out_dir, f"event_frame_{frame_count_normal:04d}.jpg")
            cv2.imwrite(image_path, frame)
            frames_since_last_event += 1

    cap.release()
    print(f"Total frames processed: {frame_count}")
    print(f"Total event-triggered frames saved: {event_frame_count}")

if __name__ == "__main__":
    video_path = 'vidToImg/Input/trial.mp4'
    output_dir = 'vidToImg/eventOutput'
    normal_out_dir = 'vidToImg/normalOutput'
    flow_threshold = 1.5
    min_frames_between_events = 30
    resize_width = None

    event_triggered_sampling_and_saving(
        video_path, 
        output_dir,
        normal_out_dir, 
        flow_threshold, 
        min_frames_between_events,
        resize_width
    )
