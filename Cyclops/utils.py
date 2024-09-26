import cv2
import os
import numpy as np
from PIL import Image
from django.conf import settings
from tempfile import NamedTemporaryFile
from .modelEvaluator import ModelEvaluator
from moviepy.editor import ImageSequenceClip



def captureFrames(video_file):
    FRAMES_DIR = os.path.join(settings.MEDIA_ROOT, 'frames')

    model_evaluator = ModelEvaluator()
    
    if not os.path.exists(FRAMES_DIR):
        os.makedirs(FRAMES_DIR)

    with NamedTemporaryFile(delete=False, suffix=".mov") as temp_video_file:
        temp_video_file.write(video_file.read())
        temp_video_file.flush()
        temp_video_file_path = temp_video_file.name

    cap = cv2.VideoCapture(temp_video_file_path)

    if not cap.isOpened():
        return {"error": "Error: Cannot open video stream"}

    frame_save_count = 1
    frame_filenames = []

    # Capture and process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_filename = f"frame_{frame_save_count}.png"
            full_frame_path = os.path.join(FRAMES_DIR, frame_filename)
            predicted_frame = model_evaluator.get_predicted_output(cropFrames(frame))
            cv2.imwrite(full_frame_path, predicted_frame)
            frame_filenames.append(full_frame_path)
            frame_save_count += 1
        else:
            break

    cap.release()

    # Delete the uploaded video after processing frames
    os.remove(temp_video_file_path)

    output_video_path = os.path.join(settings.MEDIA_ROOT, 'output_video.mp4')
    create_video_from_frames(frame_filenames, output_video_path)

    return {"message": "Frames captured successfully", "video": output_video_path}

def cropFrames(frame):
    zoom_factor = 1.5 
    top_offset = 200 
    h, w = frame.shape[:2]
    target_h = int(w * 3 / 4)
    zoomed_h = int(target_h / zoom_factor)
    zoomed_w = int(w / zoom_factor)
    start_y = top_offset
    end_y = start_y + zoomed_h
    if end_y > h:
        end_y = h
    cropped_frame = frame[start_y:end_y, (w - zoomed_w) // 2 : (w + zoomed_w) // 2]
    resized_frame = cv2.resize(np.array(cropped_frame), (640, 480))
    return (resized_frame)

def create_video_from_frames(frame_filenames, output_video_path, fps=30):
    clip = ImageSequenceClip(frame_filenames, fps=fps)
    clip.write_videofile(output_video_path, codec='libx264')