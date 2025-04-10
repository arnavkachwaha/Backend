from django.urls import path
from . import views

urlpatterns = [
    path('health/', views.health, name='health'),
    path('upload/', views.upload_form, name='upload_form'),
    path('upload_test_data/', views.upload_test_data, name='upload_test_data'),
    path('fetch_processed_video/', views.fetch_processed_video, name='fetch_processed_video'), 
]
