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
            frame = cropFrame(frame)  # Crop the frame
            cropped_frame = cv2.resize(frame, (640, 480))  # Resize the frame
            frame_filename = f"frame_{frame_save_count}.png"
            predicted_frame = model_evaluator.get_predicted_output(cropped_frame)
            full_frame_path = os.path.join(FRAMES_DIR, frame_filename)
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


def cropFrame(frame):
    img = Image.fromarray(frame)
    width, height = img.size
    left = 0
    right = width
    top = 300
    bottom = 800
    cropped_img = img.crop((left, top, right, bottom))
    
    return np.array(cropped_img)


def create_video_from_frames(frame_filenames, output_video_path, fps=30):
    clip = ImageSequenceClip(frame_filenames, fps=fps)
    clip.write_videofile(output_video_path, codec='libx264')