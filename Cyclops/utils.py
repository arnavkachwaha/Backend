import cv2
import os
import subprocess
import matplotlib
import numpy as np
import mediapipe as mp
from scipy import interpolate
from scipy.stats import zscore
import matplotlib.pyplot as plt
from django.conf import settings
from tempfile import NamedTemporaryFile
from scipy.signal import butter, filtfilt
from .modelEvaluator import ModelEvaluator
from scipy.ndimage import gaussian_filter1d 
from moviepy.editor import ImageSequenceClip

mp_face_mesh = mp.solutions.face_mesh
LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
matplotlib.use('Agg')

def processVideoForPLR(video_file):
    FRAMES_DIR = os.path.join(settings.MEDIA_ROOT, "frames", "PLR")
    fps = 30

    model_evaluator = ModelEvaluator()
    
    if not os.path.exists(FRAMES_DIR):
        os.makedirs(FRAMES_DIR)

    print('Writing video to temp file')
    with NamedTemporaryFile(delete=False, suffix=".mov") as temp_video_file:
        temp_video_file.write(video_file.read())
        temp_video_file.flush()
        temp_video_file_path = temp_video_file.name
    
    output_cropped_video_path = "./media/output_cropped_video.mp4"

    try:
        print('Cropping video')
        cap = cv2.VideoCapture(cropVideo(temp_video_file_path))

        if not cap.isOpened():
            return {"error": "Error: Cannot open video stream"}

        frame_save_count = 1
        frame_filenames = []
        frame_radius = []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        all_frames_count = 1

        # Capture and process each frame
        print('Processing frames')
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_filename = f"frame_{frame_save_count}.png"
                full_frame_path = os.path.join(FRAMES_DIR, frame_filename)
                print(f"Processing frame {all_frames_count} of {total_frames}")
                all_frames_count += 1
                predicted_frame, radius = model_evaluator.get_predicted_output(frame)
                if radius != 0:
                    frame_radius.append(radius)
                    # hard save for debug 
                    # cv2.imwrite(full_frame_path, predicted_frame)
                    # frame_filenames.append(full_frame_path)
                    # frame_save_count += 1
                    
                    # save to temp folder for prod 
                    try: 
                        with NamedTemporaryFile(delete=False, suffix=".png") as temp_frame:
                            cv2.imwrite(temp_frame.name, predicted_frame)
                            frame_filenames.append(temp_frame.name)
                            frame_save_count += 1
                    except Exception as e:
                        print(f"Error saving frame: {e}")
            else:
                break

        cap.release()
        output_video_path = os.path.join(settings.MEDIA_ROOT, "PLR", "output_video.mp4")
        print('Plotting graph')
        output_graph_path = plot_radius_over_time(frame_radius, getPlrMetrics(frame_radius), fps, video_file)
        print('Creating video')
        create_video_from_frames(frame_filenames, output_video_path)
        return {"message": "Frames captured successfully", "video": output_video_path, "graph": output_graph_path}
    
    finally:
        os.remove(output_cropped_video_path)
        os.remove(temp_video_file_path)

def cropVideo(input_path, zoom_factor=1.5, top_offset=200, output_path="./media/output_cropped_video.mp4"):
    # get the video properties
    cap = cv2.VideoCapture(input_path)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    # Calculate crop parameters
    target_h = int(w * 3 / 4)
    zoomed_h = int(target_h / zoom_factor)
    zoomed_w = int(w / zoom_factor)
    start_y = top_offset

    crop_params = f"crop={zoomed_w}:{zoomed_h}:{(w - zoomed_w) // 2}:{start_y}"

    ffmpeg_command = [
        "ffmpeg",
        "-i", input_path,          
        "-vf", crop_params, 
        '-an',  # Remove the audio stream          
        "-s", "640x480",            
        "-y",             
        '-preset', 'fast',  # Faster encoding
        '-crf', '23',  # Constant rate factor for quality (23 is a good balance)             
        output_path          
    ]

    subprocess.run(ffmpeg_command, check=True)
    return output_path

def getPlrMetrics(frame_radius):
    flashPoint = 30
    maxPD = max(frame_radius[0:flashPoint])
    minPD = min(frame_radius)
    maxPDIndex = frame_radius.index(maxPD)
    minPDIndex = frame_radius.index(minPD)
    MCV =  (maxPD - minPD) / (minPDIndex - maxPDIndex)
    _75PerOfMaxPDIndex= next((i for i in range(minPDIndex + 1, len(frame_radius)) if frame_radius[i] >= (maxPD * 0.75)),0)
    _75PerOfMaxPD = frame_radius[_75PerOfMaxPDIndex]
    PRT   = _75PerOfMaxPDIndex - minPDIndex
    Latency = next((i for i in range(flashPoint, minPDIndex) if frame_radius[i] < maxPD - 1),0) - flashPoint
    return [maxPD, minPD, MCV, _75PerOfMaxPD, PRT, Latency]

def plot_radius_over_time(frame_radius, plrMetrics, fps, video_file):
    maxPD, minPD, MCV, _75PerOfMaxPD, PRT, Latency = plrMetrics
    # Remove outliers from the frame_radius array
    filtered_radius = remove_outliers(frame_radius)

    # Apply lowpass filter to the cleaned radius data
    lowpass_filtered_radius = apply_lowpass_filter(filtered_radius, cutoff = 2 , fs=fps)

    time_values = [i / fps for i in range(len(lowpass_filtered_radius))]
    tck = interpolate.splrep(time_values, lowpass_filtered_radius, s = 1 )
    smoothed_radius_spline = interpolate.splev(time_values, tck)

    # Plotting the smoothed spline result without markers
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Move the plot to the left, leaving space for the box on the right
    plt.subplots_adjust(left=0.1, right=0.75)  
    
    ax.plot(time_values, smoothed_radius_spline, linestyle='-', color='b', label='PLR')

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Radius')
    ax.set_title('Radius Over Time')
    ax.grid(True)
    ax.legend()

    # Create a box on the right side of the plot with the PLR metrics
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    metrics_text = f"maxPD: {maxPD}\nminPD: {minPD}\nMCV: {MCV:.3f}\n75% of maxPD: {_75PerOfMaxPD}\nPRT: {PRT}\nLatency: {Latency}"

    # Add text box on the right side of the plot, moving it further to the right
    fig.text(0.8, 0.5, metrics_text, fontsize=10, bbox=props)  

    output_graph_path = os.path.join(settings.MEDIA_ROOT, "PLR", "output_graph.png")
    plt.savefig(output_graph_path, format='png')
    plt.close()

    return output_graph_path

def remove_outliers(data, threshold=2.5):
    z_scores = zscore(data)
    return np.array([x for x, z in zip(data, z_scores) if abs(z) < threshold])

def apply_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def create_video_from_frames(frame_filenames, output_video_path, fps=30):
    clip = ImageSequenceClip(frame_filenames, fps=fps)
    clip.write_videofile(output_video_path, codec='libx264')

def processVideoForVOMS(video_file):
    FRAMES_DIR = os.path.join(settings.MEDIA_ROOT, "frames", "VOMS")

    if not os.path.exists(FRAMES_DIR):
        os.makedirs(FRAMES_DIR)

    with NamedTemporaryFile(delete=False, suffix=".mov") as temp_video_file:
        temp_video_file.write(video_file.read())
        temp_video_file.flush()
        temp_video_file_path = temp_video_file.name

    frame_filenames = []
    left_eye_positions = [] 
    right_eye_positions = []
    frame_times = []
    frame_count = 0

    try:
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:

            cap = cv2.VideoCapture(temp_video_file_path)

            if not cap.isOpened():
                return {"error": "Error: Cannot open video stream"}

            frame_save_count = 1
            fps = cap.get(cv2.CAP_PROP_FPS)  # Get frame rate for time calculation

            # Capture and process each frame
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb_frame)

                    if results.multi_face_landmarks:
                        mesh_points = np.array([[int(p.x * frame.shape[1]), int(p.y * frame.shape[0])] 
                                                 for p in results.multi_face_landmarks[0].landmark])

                        (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
                        (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])

                        center_left = np.array([l_cx, l_cy], dtype=np.int32)
                        center_right = np.array([r_cx, r_cy], dtype=np.int32)

                        left_eye_positions.append(center_left[0]) 
                        right_eye_positions.append(center_right[0])

                        frame_times.append(frame_count / fps)
                        frame_count += 1

                        cv2.circle(frame, center_left, int(l_radius), (0, 255, 0), 2)
                        cv2.circle(frame, center_right, int(r_radius), (0, 255, 0), 2)

                    frame_filename = f"frame_{frame_save_count}.png"
                    full_frame_path = os.path.join(FRAMES_DIR,  frame_filename)
                    cv2.imwrite(full_frame_path, frame)
                    frame_filenames.append(full_frame_path)
                    frame_save_count += 1

                else:
                    break

        cap.release()
        
        output_video_path = os.path.join(settings.MEDIA_ROOT, "VOMS", "output_video.mp4")
        output_graph_path = os.path.join(settings.MEDIA_ROOT, "VOMS", "output_graph.png")

        plot_eye_movement_graph(left_eye_positions, right_eye_positions, frame_times, output_graph_path)
        create_video_from_frames(frame_filenames, output_video_path)

        return {"message": "Frames captured successfully", "video": output_video_path,  "graph": output_graph_path}

    finally:
        os.remove(temp_video_file_path)

def plot_eye_movement_graph(left_eye_positions, right_eye_positions, frame_times, output_graph_path):
    left_x_smooth = gaussian_filter1d(left_eye_positions, sigma=2)
    right_x_smooth = gaussian_filter1d(right_eye_positions, sigma=2)

    plt.figure(figsize=(10, 6))

    plt.plot(frame_times, left_x_smooth, label="Left Eye X", color="blue", linestyle="--")
    plt.plot(frame_times, right_x_smooth, label="Right Eye X", color="red", linestyle="-")

    plt.title("Horizontal Eye Movement (X-axis) Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Horizontal Position")
    plt.legend()
    plt.savefig(output_graph_path)
    plt.close()