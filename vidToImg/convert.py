import cv2
import os

def extract_frames(video_path, output_folder, interval_seconds=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_capture = cv2.VideoCapture(video_path)

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    interval_frames = int(fps * interval_seconds)

    frame_count = 0
    saved_frame_count = 0

    while True:
        success, frame = video_capture.read()

        if not success:
            break

        if frame_count % interval_frames == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved {frame_filename}")
            saved_frame_count += 1

        frame_count += 1

    video_capture.release()
    print("Frame extraction completed.")

extract_frames('Input/trial.mp4', 'Output', 5)
