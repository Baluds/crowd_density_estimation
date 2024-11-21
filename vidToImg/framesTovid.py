import cv2
import os
import re
import argparse
from typing import List, Tuple

def extract_frame_number(filename: str) -> int:
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"No frame number found in filename: {filename}")

def list_frames(directory: str) -> List[Tuple[str, int]]:
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    frames = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(valid_extensions):
            try:
                frame_number = extract_frame_number(filename)
                file_path = os.path.join(directory, filename)
                frames.append((file_path, frame_number))
            except ValueError as e:
                print(f"Warning: {e}. Skipping file.")
    return frames

def merge_and_sort_frames(dir1: str, dir2: str) -> List[str]:
    frames1 = list_frames(dir1)
    frames2 = list_frames(dir2)

    combined_frames = frames1 + frames2

    frame_dict = {}
    for file_path, frame_number in combined_frames:
        if frame_number in frame_dict:
            print(f"Warning: Duplicate frame number {frame_number} found in {file_path}. Overwriting previous frame.")
        frame_dict[frame_number] = file_path

    sorted_frame_numbers = sorted(frame_dict.keys())

    sorted_frames = [frame_dict[number] for number in sorted_frame_numbers]

    print(f"Total frames to merge: {len(sorted_frames)}")
    return sorted_frames

def create_video_from_frames(frames: List[str], output_video_path: str, fps: int = 30, resize_width: int = None, resize_height: int = None):
    if not frames:
        print("No frames to create a video.")
        return

    first_frame = cv2.imread(frames[0])
    if first_frame is None:
        print(f"Error: Could not read the first frame: {frames[0]}")
        return

    height, width, layers = first_frame.shape
    size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For MP4 files
    out = cv2.VideoWriter(output_video_path, fourcc, fps, size)

    out.write(first_frame)
    print(f"Writing frame 1: {frames[0]}")

    for idx, frame_path in enumerate(frames[1:], start=2):
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}. Skipping.")
            continue

        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, size)

        out.write(frame)
        if idx <= 5 or idx % 50 == 0 or idx == len(frames):
            print(f"Writing frame {idx}: {frame_path}")

    out.release()
    print(f"Video saved to {output_video_path}")

def main():

    dir1='vidToImg/normalOutput'
    dir2='vidToImg/eventOutput'
    output_video_path='vidToImg/merged_video.mp4'
    fps=24
    resize_width=None
    resize_height=None


    sorted_frames = merge_and_sort_frames(dir1, dir2)


    create_video_from_frames(sorted_frames, output_video_path, fps, resize_width, resize_height)

if __name__ == "__main__":
    main()