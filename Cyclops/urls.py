from django.urls import path
from . import views

urlpatterns = [
    path('health/', views.health, name='health'),
    path('upload/', views.upload_form, name='upload_form'),
    path('fetch_processed_video/', views.fetch_processed_video, name='fetch_processed_video'), 
]
