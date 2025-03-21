from django. db import models
import uuid

class Eyes (models.Model):
    videofile = models.FileField(upload_to='./') 

class PLRResult(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    timestamp = models.DateTimeField(auto_now_add=True)  # When created
    updated_at = models.DateTimeField(auto_now=True)  # When updated

    video_file = models.FileField(upload_to="processed_videos/")
    graph_file = models.FileField(upload_to="processed_graphs/")
    frame_diameter = models.JSONField()
    smoothed_diameter_spline = models.JSONField()

    # PLR Metrics
    maxPD = models.FloatField(null=True, blank=True)
    minPD = models.FloatField(null=True, blank=True)
    latency = models.FloatField(null=True, blank=True)
    max_constriction = models.FloatField(null=True, blank=True)
    acv = models.FloatField(null=True, blank=True)
    _75PerOfMaxPD = models.FloatField(null=True, blank=True)
    adv = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"PLRResult - {self.id} - {self.timestamp}"
