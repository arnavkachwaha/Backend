from django.contrib import admin
from .models import Eyes, PLRResult

# Register your models here.
@admin.register(Eyes)
class EyesAdmin(admin.ModelAdmin):
    list_display = ('id', 'videofile')
    search_fields = ('videofile',)
    list_filter = ('videofile',)
    ordering = ('id',)
    list_per_page = 20

@admin.register(PLRResult)
class PLRResultAdmin(admin.ModelAdmin):
    list_display = ('id', 'timestamp', 'video_file', 'graph_file')
    search_fields = ('video_file', 'graph_file')
    list_filter = ('timestamp',)
    ordering = ('-timestamp',)
    list_per_page = 20


