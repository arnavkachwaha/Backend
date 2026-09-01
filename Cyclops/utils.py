import cv2
import os
import shutil
import matplotlib
import numpy as np
import mediapipe as mp
from scipy import interpolate
from scipy.stats import zscore
import matplotlib.pyplot as plt
from django.conf import settings
from tempfile import NamedTemporaryFile
from .yoloEvaluator import YOLOEvaluator
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import datetime
from scipy.signal import butter, filtfilt, detrend
from .models import PLRResult

mp_face_mesh = mp.solutions.face_mesh
matplotlib.use('Agg')

def processVideoForPLR(video_file):
    FRAMES_DIR = os.path.join(settings.MEDIA_ROOT, "frames", "PLR")
    model_evaluator = YOLOEvaluator()
    
    if not os.path.exists(FRAMES_DIR):
        os.makedirs(FRAMES_DIR)
        
    # --- COMMENTED OUT: We do not want to save to the media/inputs folder ---
    # temp_video_file_path, timestamp = save_inputs(video_file, "PLR")

    # Instead, securely hold the input video in a temporary system file
    temp_input = NamedTemporaryFile(delete=False, suffix=".mp4")
    for chunk in video_file.chunks():
        temp_input.write(chunk)
    temp_input.close() # Close so OpenCV can access it safely
    
    temp_video_file_path = temp_input.name
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    temp_video_path = None
    temp_graph_path = None
    frame_filenames = []

    try:
        cap = cv2.VideoCapture(temp_video_file_path)

        if not cap.isOpened():
            return {"error": "Error: Cannot open video stream"}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_save_count = 1
        frame_diameter = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print('Processing frames')
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                try: 
                    frame = cropFrame(frame)
                    predicted_frame, diameter = model_evaluator.get_predicted_output(frame)
                    
                    if diameter != 0:
                        frame_diameter.append(diameter)
                        with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_frame:
                            cv2.imwrite(temp_frame.name, predicted_frame)
                            frame_filenames.append(temp_frame.name)
                    elif diameter == 0 and frame_diameter:
                        frame_diameter.append(frame_diameter[-1])
                        last_frame_path = frame_filenames[-1]
                        with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_frame:
                            shutil.copy(last_frame_path, temp_frame.name)
                            frame_filenames.append(temp_frame.name)
                    frame_save_count += 1
                except Exception as e:
                    print(f"Error saving frame: {e}")
            else:
                break

        cap.release()

        print('Smoothing diameter data')
        smoothed_diameter_spline, time_values = get_smoothed_diameter_spline(frame_diameter, fps)
        
        print('Calculating PLR metrics')
        plr_metrics = getPlrMetrics(smoothed_diameter_spline, fps)

        print('Plotting graph')
        temp_graph_path = plot_diameter_over_time(smoothed_diameter_spline, time_values, plr_metrics)

        print('Creating video')
        temp_video_path = create_video_from_frames(frame_filenames, fps)
        vid_file, graph_file = save_outputs(temp_video_path, temp_graph_path, "PLR", timestamp)

        maxPD, minPD, latency, max_constriction, acv, _75PerOfMaxPD, adv = plr_metrics
        plr_result = PLRResult.objects.create(
            video_file=vid_file,
            graph_file=graph_file,
            frame_diameter=frame_diameter,
            smoothed_diameter_spline=smoothed_diameter_spline,
            maxPD=maxPD,
            minPD=minPD,
            latency=latency,
            max_constriction=max_constriction,
            acv=acv,
            _75PerOfMaxPD=_75PerOfMaxPD,
            adv=adv,
        )

        metrics = {
            "maxPD": maxPD,
            "minPD": minPD,
            "latency": latency,
            "max_constriction": max_constriction,
            "acv": acv,
            "_75PerOfMaxPD": _75PerOfMaxPD,
            "adv": adv
        }

        return {"message": "Frames captured successfully", "video": vid_file, "graph": graph_file, "metrics": metrics}
    
    finally:
        # ABSOLUTE CLEANUP: Destroy all temporary generation files & the input video immediately
        if temp_video_file_path and os.path.exists(temp_video_file_path):
            os.remove(temp_video_file_path)
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if temp_graph_path and os.path.exists(temp_graph_path):
            os.remove(temp_graph_path)
        for frame in frame_filenames:
            if os.path.exists(frame):
                os.remove(frame)
        

def save_outputs(temp_video_path, temp_graph_path, output_type, timestamp=False):
    try:
        with open(temp_video_path, 'rb') as video_file, open(temp_graph_path, 'rb') as graph_file:
            video_data = video_file.read()
            graph_data = graph_file.read()

            if not timestamp:
                timestamp  = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            
            vid_dir = os.path.join(settings.MEDIA_ROOT, output_type, "outputs", "videos")
            graph_dir = os.path.join(settings.MEDIA_ROOT, output_type, "outputs", "graphs")

            if not os.path.exists(vid_dir):
                os.makedirs(vid_dir)
            if not os.path.exists(graph_dir):
                os.makedirs(graph_dir)

            vid_path = os.path.join(vid_dir, f"{timestamp}.mp4")
            graph_path = os.path.join(graph_dir, f"{timestamp}.jpg")

            with open(vid_path, 'wb') as output_video_file:
                output_video_file.write(video_data)

            with open(graph_path, 'wb') as output_graph_file:
                output_graph_file.write(graph_data)
        
        return vid_path, graph_path

    except Exception as e:
        print(f"Error saving outputs: {e}")
        return None, None

def save_inputs(video_file, output_type):
    try:
        timestamp  = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        video_dir = os.path.join(settings.MEDIA_ROOT, output_type, "inputs")

        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        video_path = os.path.join(video_dir, f"{timestamp}.mp4")

        with open(video_path, 'wb') as output_video_file:
            output_video_file.write(video_file.read())

        return video_path, timestamp
        
    except Exception as e:
        print(f"Error saving inputs: {e}")
        return None, None

def cropFrame(frame):
    if frame.shape[0] == 640 and frame.shape[1] == 640:
        return frame
        
    print('Cropping frame to 640x640 from top')
    h, w = frame.shape[:2]
    crop_size = 640
    start_x = max(0, (w - crop_size) // 2)
    end_x = min(w, start_x + crop_size)
    top_offset = 100 
    start_y = min(top_offset, h - crop_size)
    end_y = start_y + crop_size
    cropped_frame = frame[start_y:end_y, start_x:end_x]
    resized_frame = cv2.resize(cropped_frame, (640, 640))
    return resized_frame

def getPlrMetrics(frame_diameter, fps):
    print('Calculating PLR metrics')
    if not frame_diameter or fps <= 0:
        return [0, 0, 0, 0, 0, 0, 0]

    # Calculate the flash frame dynamically based on a 1.0 second delay and the video FPS
    flashPoint = int(fps * 1.0)
    
    if len(frame_diameter) <= flashPoint:
        flashPoint = 0

    pre_flash_slice = frame_diameter[0:max(1, flashPoint)]
    post_flash_slice = frame_diameter[flashPoint:] if flashPoint < len(frame_diameter) else frame_diameter

    maxPD = max(pre_flash_slice)
    maxPDIndex = frame_diameter.index(maxPD)

    minPD = min(post_flash_slice)
    minPDIndex = flashPoint + post_flash_slice.index(minPD)

    # 1. FIX LATENCY: Find exactly when it drops below 95% after the flash.
    constriction_target = maxPD * 0.95
    try:
        constriction_onset_index = next(
            i for i in range(flashPoint, len(frame_diameter)) if frame_diameter[i] <= constriction_target
        )
        latency = round(max(0, (constriction_onset_index - flashPoint)) / fps, 4)
    except StopIteration:
        constriction_onset_index = maxPDIndex # Fallback for ACV math
        latency = -1.0 # -1.0 means it failed to detect the drop

    # 2. FIX ACV: (End Size - Start Size) / Constriction Time (Results in a negative value)
    max_constriction = np.round((maxPD - minPD), 2)
    constriction_time = max(1 / fps, abs(minPDIndex - constriction_onset_index) / fps)
    acv = round((minPD - maxPD) / constriction_time, 2)

    # 3. FIX T75: Find when it recovers 75% of the constricted amount.
    target_75_recovery = minPD + (0.75 * (maxPD - minPD))
    try:
        _75PerIndex = next(
            i for i in range(minPDIndex, len(frame_diameter)) if frame_diameter[i] >= target_75_recovery
        )
        _75PerOfMaxPD = np.round(max(0, (_75PerIndex - minPDIndex)) / fps, 2)
    except StopIteration:
        _75PerOfMaxPD = -1.0 # -1.0 means the video ended before recovery

    # ADV
    post_min_slice = frame_diameter[minPDIndex:]
    if len(post_min_slice) > 0:
        max_dilated_diameter = max(post_min_slice)
        max_dilated_index = minPDIndex + post_min_slice.index(max_dilated_diameter)
        max_dilation = max_dilated_diameter - minPD
        max_dilation_time = max(1 / fps, (max_dilated_index - minPDIndex) / fps)
        adv = round(max_dilation / max_dilation_time, 2)
    else:
        adv = 0.0

    return [maxPD, minPD, latency, max_constriction, acv, _75PerOfMaxPD, adv]

def plot_diameter_over_time(smoothed_diameter_spline, time_values, plr_metrics):
    maxPD, minPD, latency, max_constriction, acv, _75PerOfMaxPD, adv = plr_metrics

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.1, right=0.75)  
    
    ax.plot(time_values, smoothed_diameter_spline, linestyle='-', color='b', label='PLR')

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('diameter')
    ax.set_title('diameter Over Time')
    ax.grid(True)
    ax.legend()

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    metrics_text = f"""
    maxPD: {maxPD}mm
    minPD: {minPD}mm
    Delta: {max_constriction}mm
    Latency: {latency}s
    T75%: {_75PerOfMaxPD}s
    ACV: {acv:.3f}mm/s
    ADV: {adv:.3f}mm/s
    """

    fig.text(0.8, 0.5, metrics_text, fontsize=12, bbox=props)  

    with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_graph_file:
        plt.savefig(temp_graph_file.name)
        plt.close()
        return temp_graph_file.name

def get_smoothed_diameter_spline(frame_diameter, fps):
    # Guard: if no frames detected or too few frames, bypass interpolation to prevent m>k errors
    if not frame_diameter or len(frame_diameter) <= 3:
        time_values = [i / fps for i in range(len(frame_diameter))] if fps > 0 else []
        return frame_diameter, time_values

    filtered_diameter = remove_outliers(frame_diameter)
    lowpass_filtered_diameter = apply_lowpass_filter(filtered_diameter, cutoff = 2 , fs=fps)

    time_values = [i / fps for i in range(len(lowpass_filtered_diameter))]
    
    # Guard: SciPy spline requires at least 4 points (m > k, where k=3 is default)
    if len(lowpass_filtered_diameter) <= 3:
        return np.round(lowpass_filtered_diameter, 2).tolist(), time_values

    tck = interpolate.splrep(time_values, lowpass_filtered_diameter, s = 1 )
    smoothed_diameter_spline = interpolate.splev(time_values, tck)   

    smoothed_diameter_spline = np.round(smoothed_diameter_spline, 2) 

    return smoothed_diameter_spline.tolist(), time_values

def remove_outliers(data, threshold=2.5):
    z_scores = zscore(data)
    cleaned_data = np.array(data, dtype=float)
    
    for i, z in enumerate(z_scores):
        if abs(z) >= threshold:
            cleaned_data[i] = np.nan
            
    nans = np.isnan(cleaned_data)
    
    if np.any(nans) and not np.all(nans):
        x = lambda z: z.nonzero()[0]
        cleaned_data[nans] = np.interp(x(nans), x(~nans), cleaned_data[~nans])
    elif np.all(nans):
        cleaned_data = np.zeros_like(cleaned_data)
        
    return cleaned_data

def apply_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Safe guard for short arrays to prevent SciPy 'padlen' crash
    if len(data) <= 18:
        return data
        
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def create_video_from_frames(frame_filenames, fps=30):
    clip = ImageSequenceClip(frame_filenames, fps=fps)
    with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output_video_file:
        clip.write_videofile(temp_output_video_file.name, codec='libx264')
        return temp_output_video_file.name
    return ""

def processVideoForVOMS(video_file):
    print('Processing video for VOMS') 
    FRAMES_DIR = os.path.join(settings.MEDIA_ROOT, "frames", "VOMS")
    if not os.path.exists(FRAMES_DIR):
        os.makedirs(FRAMES_DIR)

    temp_video_file_path, timestamp = save_inputs(video_file, "VOMS")
    
    temp_graph_file = None
    temp_video_path = None
    frame_filenames = []

    try:
        with mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:

            extreme_points = {
                "Left Iris Left Extreme": 474,
                "Left Iris Right Extreme": 476,
                "Right Iris Left Extreme": 469,
                "Right Iris Right Extreme": 471
            }

            cap = cv2.VideoCapture(temp_video_file_path)

            if not cap.isOpened():
                return {"error": "Error: Cannot open video stream"}
            
            left_iris_centers = []
            right_iris_centers = []

            frame_save_count = 1
            fps = cap.get(cv2.CAP_PROP_FPS)
            time_per_frame = 1000 / fps 

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                print(f"Processing frame {frame_save_count}")

                results = face_mesh.process(frame)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        left_iris_left = face_landmarks.landmark[extreme_points["Left Iris Left Extreme"]]
                        left_iris_right = face_landmarks.landmark[extreme_points["Left Iris Right Extreme"]]
                        l_cx = (left_iris_left.x + left_iris_right.x) / 2
                        l_cy = (left_iris_left.y + left_iris_right.y) / 2
                        l_radius = np.sqrt((left_iris_left.x - left_iris_right.x) ** 2 + 
                                           (left_iris_left.y - left_iris_right.y) ** 2) / 2

                        l_cx, l_cy = int(l_cx * frame.shape[1]), int(l_cy * frame.shape[0])
                        l_radius = int(l_radius * frame.shape[1])

                        right_iris_left = face_landmarks.landmark[extreme_points["Right Iris Left Extreme"]]
                        right_iris_right = face_landmarks.landmark[extreme_points["Right Iris Right Extreme"]]
                        r_cx = (right_iris_left.x + right_iris_right.x) / 2
                        r_cy = (right_iris_left.y + right_iris_right.y) / 2
                        r_radius = np.sqrt((right_iris_left.x - right_iris_right.x) ** 2 + 
                                           (right_iris_left.y - right_iris_right.y) ** 2) / 2

                        r_cx, r_cy = int(r_cx * frame.shape[1]), int(r_cy * frame.shape[0])
                        r_radius = int(r_radius * frame.shape[1])

                        cv2.circle(frame, (l_cx, l_cy), l_radius, (0, 255, 0), 2)
                        cv2.circle(frame, (r_cx, r_cy), r_radius, (0, 255, 0), 2)

                        left_iris_centers.append((l_cx, l_cy))
                        right_iris_centers.append((r_cx, r_cy))

                    frame_filename = f"frame_{frame_save_count}.jpg"
                    full_frame_path = os.path.join(FRAMES_DIR, frame_filename)
                    cv2.imwrite(full_frame_path, frame)
                    frame_filenames.append(full_frame_path)
                    frame_save_count += 1

            cap.release()

            times = np.arange(0, len(left_iris_centers)) * time_per_frame

            temp_graph_file = plot_iris_center_graph(times, left_iris_centers, right_iris_centers)
            temp_video_path = create_video_from_frames(frame_filenames, fps)
            
            video_file, graph_file = save_outputs(temp_video_path, temp_graph_file, "VOMS", timestamp)

            return {"message": "Frames captured successfully", "video": video_file,  "graph": graph_file}

    finally:
        # ABSOLUTE CLEANUP: Destroy temporary files AND the initial input video immediately
        if temp_video_file_path and os.path.exists(temp_video_file_path):
            os.remove(temp_video_file_path)
        if temp_graph_file and os.path.exists(temp_graph_file):
            os.remove(temp_graph_file)
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        for frame in frame_filenames:
            if os.path.exists(frame):
                os.remove(frame)

def plot_iris_center_graph(times, left_iris_centers, right_iris_centers):
    print('Plotting graph')
    left_x = [pos[0] for pos in left_iris_centers]
    right_x = [pos[0] for pos in right_iris_centers]

    left_x_detrended = detrend(left_x - np.mean(left_x))
    right_x_detrended = detrend(right_x - np.mean(right_x))

    window_size = 15  
    left_x_smoothed = np.convolve(left_x_detrended, np.ones(window_size)/window_size, mode='same')
    right_x_smoothed = np.convolve(right_x_detrended, np.ones(window_size)/window_size, mode='same')

    plt.figure(figsize=(10, 6))
    plt.plot(times, left_x_smoothed, color="red", label="Left Eye Iris Center X Position")
    plt.plot(times, right_x_smoothed, color="blue", label="Right Eye Iris Center X Position")
    
    plt.ylim(-50, 50)  
    
    with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_graph_file:
        plt.title("Horizontal Eye Movement (X-axis) Over Time")
        plt.xlabel("Time (milliseconds)")
        plt.ylabel("Horizontal Position (Normalized)")
        plt.legend()
        plt.grid(True)
        plt.savefig(temp_graph_file.name)
        plt.close()

        return temp_graph_file.name