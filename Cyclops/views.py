import json
import os
import threading
from django.conf import settings
from django.shortcuts import render
from Cyclops.forms import VideoForm
from Cyclops.models import TestResult
from Cyclops.utils import processVideoForPLR , processVideoForVOMS
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt

def delayed_cleanup(file_paths):
    """Background task to delete files from the disk after a delay."""
    for path in file_paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
                print(f"Successfully cleaned up space: {path}")
            except Exception as e:
                print(f"Cleanup error: {e}")

@csrf_exempt
def health(request):
    return JsonResponse({'status': 'OK'}, status=200)

@csrf_exempt
def upload_form(request):
    if request.method == 'POST':
        videoType = request.POST.get('videoType')
        form = VideoForm(request.POST, request.FILES)
        
        if form.is_valid():
            videoFile = request.FILES['videofile']
            try:
                if videoType == "VOMS":
                    print("Processing video for VOMS")
                    result = processVideoForVOMS(videoFile)
                else:
                    print("Processing video for PLR")
                    result = processVideoForPLR(videoFile)
                    
                if 'error' in result:
                    return JsonResponse({'message': result['error']}, status=500)
                
                video_path = result.get('video')
                graph_path = result.get('graph')
                metrics = result.get('metrics')

                if video_path and os.path.exists(video_path) and graph_path and os.path.exists(graph_path):
                    video_download_url = request.build_absolute_uri(f'/media/{videoType}/outputs/videos/{os.path.basename(video_path)}')
                    graph_download_url = request.build_absolute_uri(f'/media/{videoType}/outputs/graphs/{os.path.basename(graph_path)}')

                    # Trigger a background thread to delete the outputs in 60 seconds
                    # This allows the frontend enough time to hit the download URLs
                    threading.Timer(60.0, delayed_cleanup, args=([video_path, graph_path],)).start()

                    return JsonResponse({
                        'video_download_url': video_download_url,
                        'graph_download_url': graph_download_url,
                        'metrics': metrics
                    }, status=200)

            except Exception as e:
                return JsonResponse({'message': str(e)}, status=500)
            else:
                return JsonResponse({"message": "Video processing failed"}, status=500)
        else:
            return JsonResponse({'message': 'Form is invalid', 'errors': form.errors}, status=400)

    elif request.method == 'GET':
        return render(request, 'upload.html', {'form': VideoForm()})
    
    return JsonResponse({'message': 'Invalid request method'}, status=405)


def fetch_processed_video(request):
    if request.method == 'GET':
        media_root = settings.MEDIA_ROOT
        video_filename = 'output_video.mp4'
        video_path = os.path.join(media_root, video_filename)

        if os.path.exists(video_path):
            with open(video_path, 'rb') as f:
                response = HttpResponse(f.read(), content_type='video/mp4')
                response['Content-Disposition'] = f'inline; filename="{video_filename}"'
                return response
        else:
            return JsonResponse({'message': 'Processed video not found'}, status=404)
    
    return JsonResponse({'message': 'Invalid request method'}, status=405)

@csrf_exempt
def upload_test_data(request):
    # Keep exactly as you have it
    if request.method == 'POST':
        video_file = request.FILES.get('videofile')
        test_type = request.POST.get('testType')
        plot_data_str = request.POST.get('plotData')
        fps_str = request.POST.get('fps', '30')
        maxPD_str = request.POST.get('maxPD', '')
        minPD_str = request.POST.get('minPD', '')
        latency = request.POST.get('latency', '')
        max_constriction_str = request.POST.get('maxConstriction', '')
        seventyFivePercentRecovery = request.POST.get('seventyFivePercentRecovery', '')
        adv_str = request.POST.get('adv', '')
        acv_str = request.POST.get('acv', '')
        iris_bounding_boxes_str = request.POST.get('irisBoundingBoxes', '')
        pupil_bounding_boxes_str = request.POST.get('pupilBoundingBoxes', '')
        
        try:
            plot_data = json.loads(plot_data_str) if plot_data_str else []
        except Exception as e:
            return JsonResponse({'message': f'Invalid plotData format: {e}'}, status=400)
        
        try:
            iris_bounding_boxes = json.loads(iris_bounding_boxes_str) if iris_bounding_boxes_str else []
        except Exception as e:
            return JsonResponse({'message': f'Invalid irisData format: {e}'}, status=400)
        
        try:
            pupil_bounding_boxes = json.loads(pupil_bounding_boxes_str) if pupil_bounding_boxes_str else []
        except Exception as e:
            return JsonResponse({'message': f'Invalid pupilData format: {e}'}, status=400)
        
        if not video_file or not test_type:
            return JsonResponse({'message': 'Missing required fields: videofile and testType'}, status=400)
        
        try:
            fps = float(fps_str)
        except ValueError:
            fps = 30.0
        
        def parse_float(val):
            try:
                return float(val) if val != "" else None
            except ValueError:
                return None
        
        maxPD = parse_float(maxPD_str)
        minPD = parse_float(minPD_str)
        max_constriction = parse_float(max_constriction_str)
        adv = parse_float(adv_str)
        acv = parse_float(acv_str)
        
        try:
            test_instance = TestResult.objects.create(
                video_file=video_file,
                test_type=test_type,
                plotData=plot_data,
                fps=fps,
                maxPD=maxPD,
                minPD=minPD,
                latency=latency,
                max_constriction=max_constriction,
                seventyFivePercentRecovery=seventyFivePercentRecovery,
                adv=adv,
                acv=acv,
                iris_bounding_boxes=iris_bounding_boxes,
                pupil_bounding_boxes=pupil_bounding_boxes
            )
            return JsonResponse({
                'message': 'Test data saved successfully',
                'id': str(test_instance.id)
            }, status=200)
        except Exception as e:
            return JsonResponse({'message': str(e)}, status=500)
        
    elif request.method == 'GET':
        return HttpResponse("Upload test data", status=200)
    
    return JsonResponse({'message': 'Invalid request method'}, status=405)