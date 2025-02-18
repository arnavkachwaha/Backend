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
from moviepy.editor import ImageSequenceClip
import datetime
from scipy.signal import butter, filtfilt, detrend

mp_face_mesh = mp.solutions.face_mesh
matplotlib.use('Agg')

def processVideoForPLR(video_file):
    FRAMES_DIR = os.path.join(settings.MEDIA_ROOT, "frames", "PLR")

    model_evaluator = YOLOEvaluator()
    
    if not os.path.exists(FRAMES_DIR):
        os.makedirs(FRAMES_DIR)

    print('Writing video to temp file')
    # with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
    #     temp_video_file.write(video_file.read())
    #     temp_video_file.flush()
    #     temp_video_file_path = temp_video_file.name

    # save the input video, hard save for local data collection 
    temp_video_file_path, timestamp = save_inputs(video_file, "PLR")

    # initialize temp paths for finally block 
    temp_video_path = None
    temp_graph_path = None
    frame_filenames = []

    try:
        cap = cv2.VideoCapture(temp_video_file_path)

        if not cap.isOpened():
            return {"error": "Error: Cannot open video stream"}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_save_count = 1
        frame_filenames = []
        frame_diameter = []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        all_frames_count = 1

        # Capture and process each frame
        print('Processing frames')
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                try: 
                    mm_diameter, pupil_circle, iris_circle = model_evaluator.get_pupil_diameter(frame)
                    

                    # if diameter != 0:
                    #     frame_diameter.append(diameter)
                    #     with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_frame:
                    #         cv2.imwrite(temp_frame.name, predicted_frame)
                    #         frame_filenames.append(temp_frame.name)
                    # elif diameter == 0 and frame_diameter:
                    #     frame_diameter.append(frame_diameter[-1])
                    #     # Copy the last saved frame to the new temporary file
                    #     last_frame_path = frame_filenames[-1]
                    #     with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_frame:
                    #         shutil.copy(last_frame_path, temp_frame.name)
                    #         frame_filenames.append(temp_frame.name)
                    # frame_save_count += 1
                except Exception as e:
                    print(f"Error saving frame: {e}")
            else:
                break

        cap.release()

        print('Smoothing diameter data')
        smoothed_diameter_spline, time_values = get_smoothed_diameter_spline(frame_diameter, fps)
        
        print('Plotting graph')
        temp_graph_path = plot_diameter_over_time(smoothed_diameter_spline, time_values, fps)

        print('Creating video')
        temp_video_path = create_video_from_frames(frame_filenames, fps)
        vid_file, graph_file = save_outputs(temp_video_path, temp_graph_path, "PLR", timestamp)

        return {"message": "Frames captured successfully", "video": vid_file, "graph": graph_file}
    
    finally:
        # Ensure cleanup in the finally block
        if temp_video_path:
            os.remove(temp_video_path)
        if temp_graph_path:
            os.remove(temp_graph_path)
        for frame in frame_filenames:
            os.remove(frame)
        

def save_outputs(temp_video_path, temp_graph_path, output_type, timestamp=False):
    try:
        with open(temp_video_path, 'rb') as video_file, open(temp_graph_path, 'rb') as graph_file:
            video_data = video_file.read()
            graph_data = graph_file.read()

            # generate file_name based on date and timestamp
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
        # save the InMempryUploadedFile to media folder
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

# Calculate crop parameters
def cropFrame(frame):
    # check if already 640x480
    if frame.shape[0] == 480 and frame.shape[1] == 640:
        return frame
    print('Cropping frame')
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

def getPlrMetrics(frame_diameter, fps):
    print('Calculating PLR metrics')
    flashPoint = 35
    maxPD = max(frame_diameter[0:flashPoint])
    minPD = min(frame_diameter[flashPoint:])
# AttributeError: 'numpy.ndarray' object has no attribute 'index'
    maxPDIndex = frame_diameter.index(maxPD)
    minPDIndex = frame_diameter.index(minPD)
    _10PerAftrFlashPt = next((i for i in range(maxPDIndex, len(frame_diameter)) if frame_diameter[i] <= (maxPD * 0.90)),0)
    #####################
    max_constiction =  np.abs((maxPD - minPD))
    max_constriction_time = np.abs((frame_diameter.index(maxPD) - frame_diameter.index(minPD))) / fps 
    mcv = round(max_constiction / max_constriction_time,2)
    #####################
    _75PerOfMaxPDIndex= next((i for i in range(minPDIndex, len(frame_diameter)) if frame_diameter[i] >= (maxPD * 0.75)),0)
    _75PerOfMaxPD = str(round(np.abs(_75PerOfMaxPDIndex - minPD) / 30,2)) 
    Latency = str(round(np.abs(maxPDIndex - _10PerAftrFlashPt) / 30, 4))
    return [maxPD, minPD, mcv, _75PerOfMaxPD, Latency]

def plot_diameter_over_time(smoothed_diameter_spline, time_values, fps):
  
    maxPD, minPD, mcv, _75PerOfMaxPD, Latency = getPlrMetrics(smoothed_diameter_spline, fps)

    # Plotting the smoothed spline result without markers
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Move the plot to the left, leaving space for the box on the right
    plt.subplots_adjust(left=0.1, right=0.75)  
    
    ax.plot(time_values, smoothed_diameter_spline, linestyle='-', color='b', label='PLR')

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('diameter')
    ax.set_title('diameter Over Time')
    ax.grid(True)
    ax.legend()

    # Create a box on the right side of the plot with the PLR metrics
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    metrics_text = f"maxPD: {maxPD}mm\nminPD: {minPD}mm\nMCV: {mcv:.3f}mm/s\n75% of maxPD: {_75PerOfMaxPD}s\nLatency: {Latency}s"

    # Add text box on the right side of the plot, moving it further to the right
    fig.text(0.8, 0.5, metrics_text, fontsize=12, bbox=props)  

    # save to temp output_graph_path
    with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_graph_file:
        plt.savefig(temp_graph_file.name)
        plt.close()
        return temp_graph_file.name
    
    return ""

def get_smoothed_diameter_spline(frame_diameter, fps):
    # Remove outliers from the frame_diameter array
    filtered_diameter = remove_outliers(frame_diameter)

    # Apply lowpass filter to the cleaned diameter data
    lowpass_filtered_diameter = apply_lowpass_filter(filtered_diameter, cutoff = 2 , fs=fps)

    time_values = [i / fps for i in range(len(lowpass_filtered_diameter))]
    tck = interpolate.splrep(time_values, lowpass_filtered_diameter, s = 1 )
    smoothed_diameter_spline = interpolate.splev(time_values, tck)   

    # round to 2 decimal places
    smoothed_diameter_spline = np.round(smoothed_diameter_spline, 2) 

    return smoothed_diameter_spline.tolist(), time_values

def remove_outliers(data, threshold=2.5):
    z_scores = zscore(data)
    return np.array([x for x, z in zip(data, z_scores) if abs(z) < threshold])

def apply_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
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

    # with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
    #     temp_video_file.write(video_file.read())
    #     temp_video_file.flush()
    #     temp_video_file_path = temp_video_file.name

    # save the input video, hard save for local data collection
    temp_video_file_path, timestamp = save_inputs(video_file, "VOMS")
    
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

            frame_filenames = []
            frame_save_count = 1
            fps = cap.get(cv2.CAP_PROP_FPS)
            time_per_frame = 1000 / fps  # Time in milliseconds

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                print(f"Processing frame {frame_save_count}")

                results = face_mesh.process(frame)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Calculate left iris center and diameter
                        left_iris_left = face_landmarks.landmark[extreme_points["Left Iris Left Extreme"]]
                        left_iris_right = face_landmarks.landmark[extreme_points["Left Iris Right Extreme"]]
                        l_cx = (left_iris_left.x + left_iris_right.x) / 2
                        l_cy = (left_iris_left.y + left_iris_right.y) / 2
                        l_radius = np.sqrt((left_iris_left.x - left_iris_right.x) ** 2 + 
                                           (left_iris_left.y - left_iris_right.y) ** 2) / 2

                        # Convert to pixel space
                        l_cx, l_cy = int(l_cx * frame.shape[1]), int(l_cy * frame.shape[0])
                        l_radius = int(l_radius * frame.shape[1])

                        # Calculate right iris center and radius
                        right_iris_left = face_landmarks.landmark[extreme_points["Right Iris Left Extreme"]]
                        right_iris_right = face_landmarks.landmark[extreme_points["Right Iris Right Extreme"]]
                        r_cx = (right_iris_left.x + right_iris_right.x) / 2
                        r_cy = (right_iris_left.y + right_iris_right.y) / 2
                        r_radius = np.sqrt((right_iris_left.x - right_iris_right.x) ** 2 + 
                                           (right_iris_left.y - right_iris_right.y) ** 2) / 2

                        # Convert to pixel space
                        r_cx, r_cy = int(r_cx * frame.shape[1]), int(r_cy * frame.shape[0])
                        r_radius = int(r_radius * frame.shape[1])

                        # Draw circles around the irises
                        cv2.circle(frame, (l_cx, l_cy), l_radius, (0, 255, 0), 2)
                        cv2.circle(frame, (r_cx, r_cy), r_radius, (0, 255, 0), 2)

                        # Store iris center positions for plotting
                        left_iris_centers.append((l_cx, l_cy))
                        right_iris_centers.append((r_cx, r_cy))

                    # Save the processed frame
                    frame_filename = f"frame_{frame_save_count}.jpg"
                    full_frame_path = os.path.join(FRAMES_DIR, frame_filename)
                    cv2.imwrite(full_frame_path, frame)
                    frame_filenames.append(full_frame_path)
                    frame_save_count += 1

            cap.release()

            # Calculate times for each frame for plotting
            times = np.arange(0, len(left_iris_centers)) * time_per_frame

            temp_graph_file = plot_iris_center_graph(times, left_iris_centers, right_iris_centers)
            temp_video_path = create_video_from_frames(frame_filenames, fps)
            # save the outputs
            video_file, graph_file = save_outputs(temp_video_path, temp_graph_file, "VOMS", timestamp)

            return {"message": "Frames captured successfully", "video": video_file,  "graph": graph_file}

    finally:
        # os.remove(temp_video_file_path) hard save for local data collection
        os.remove(temp_graph_file)
        os.remove(temp_video_path)
        for frame in frame_filenames:
            os.remove(frame)

def plot_iris_center_graph(times, left_iris_centers, right_iris_centers):
    print('Plotting graph')
    # Extract the X coordinates of the iris centers
    left_x = [pos[0] for pos in left_iris_centers]
    right_x = [pos[0] for pos in right_iris_centers]

    # Detrend to remove any linear drift and normalize by subtracting the mean
    left_x_detrended = detrend(left_x - np.mean(left_x))
    right_x_detrended = detrend(right_x - np.mean(right_x))

    # Smooth the data if needed using a moving average
    window_size = 15  # Adjust the window size for smoothing as needed
    left_x_smoothed = np.convolve(left_x_detrended, np.ones(window_size)/window_size, mode='same')
    right_x_smoothed = np.convolve(right_x_detrended, np.ones(window_size)/window_size, mode='same')

    # Plot the smoothed, normalized data
    plt.figure(figsize=(10, 6))
    plt.plot(times, left_x_smoothed, color="red", label="Left Eye Iris Center X Position")
    plt.plot(times, right_x_smoothed, color="blue", label="Right Eye Iris Center X Position")
    
    # Set the y-axis limits for a better sinusoidal appearance
    plt.ylim(-50, 50)  # Adjust these values based on the data range
    
    with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_graph_file:
        plt.title("Horizontal Eye Movement (X-axis) Over Time")
        plt.xlabel("Time (milliseconds)")
        plt.ylabel("Horizontal Position (Normalized)")
        plt.legend()
        plt.grid(True)
        plt.savefig(temp_graph_file.name)
        plt.close()

        return temp_graph_file.name
    
    return ""
 
    def overlay_mask_on_image(image, pupil_circle, iris_circle):
     # Draw the circle onto the blended image
        cv2.circle(image, pupil_center, pupil_pixel_radius, (0, 255, 0), 2)  # Green circle
        cv2.putText(image, str(mm_diameter), 
                    org = (pupil_center[0] + pupil_pixel_radius + 1, pupil_center[1] + pupil_pixel_radius + 1), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 1, 
                    color = (0, 255, 0), 
                    thickness = 2)

        # draw iris bounding box
        cv2.rectangle(image, (int(iris_circle[0][0] - iris_circle[1]), int(iris_circle[0][1] - iris_circle[1])), (int(iris_circle[0][0] + iris_circle[1]), int(iris_circle[0][1] + iris_circle[1])), (255, 255, 255), 1)