import os
from django.conf import settings
from django.shortcuts import render
from Cyclops.forms import VideoForm
from Cyclops.utils import captureFrames
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def upload_form(request):
    if request.method == 'POST':
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            video_file = request.FILES['videofile']
            result = captureFrames(video_file)
            if 'error' in result:
                return JsonResponse({'message': result['error']}, status=400)

            # Serve the processed video as a response
            video_path = result.get('video')
            if video_path and os.path.exists(video_path):
                with open(video_path, 'rb') as f:
                    response = HttpResponse(f.read(), content_type='video/mp4')
                    response['Content-Disposition'] = f'inline; filename="output_video.mp4"'
                    return response
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
