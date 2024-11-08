import os
from django.conf import settings
from django.shortcuts import render
from Cyclops.forms import VideoForm
from Cyclops.utils import processVideoForPLR , processVideoForVOMS
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt

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

                if video_path and os.path.exists(video_path) and graph_path and os.path.exists(graph_path):
                    video_download_url = request.build_absolute_uri(f'/media/{videoType}/outputs/videos/{os.path.basename(video_path)}')
                    graph_download_url = request.build_absolute_uri(f'/media/{videoType}/outputs/graphs/{os.path.basename(graph_path)}')

                    return JsonResponse({
                        'video_download_url': video_download_url,
                        'graph_download_url': graph_download_url
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
