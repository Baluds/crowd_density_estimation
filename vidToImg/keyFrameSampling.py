import cv2
from skimage.metrics import structural_similarity as ssim
import os

def is_keyframe(frame1, frame2, threshold=0.5):
    threshold = float(threshold)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score < threshold

def keyframe_sampling(video_path, output_dir, threshold=0.5):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    keyframes = []

    if not ret:
        print("Could not read the video.")
        return
    
    frame_count = 0
    keyframe_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if is_keyframe(prev_frame, frame, threshold):
            keyframe_count += 1
            
            image_path = os.path.join(output_dir, f"keyframe_{keyframe_count}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Saved {image_path}")
            prev_frame = frame  

    cap.release()
    print(f"Extracted and saved {keyframe_count} keyframes.")


video_path = 'vidToImg/Input/trial.mp4'
output_dir = 'vidToImg/Output'
keyframe_sampling(video_path, output_dir, 0.5)