from django. db import models
import uuid

class Eyes (models.Model):
    videofile = models.FileField(upload_to='./') 

def test_video_upload_path(instance, filename):
    # instance.test_type is expected to contain the test type, e.g. "PLR" or "VOMS"
    return f"recorded_videos/{instance.test_type}/{filename}"

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

class TestResult(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    test_type = models.CharField(max_length=50)  # e.g., "PLR" or "VOMS"
    timestamp = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    video_file = models.FileField(upload_to=test_video_upload_path)
    plotData = models.JSONField()
    fps = models.FloatField(null=True, blank=True)
    
    maxPD = models.FloatField(null=True, blank=True)
    minPD = models.FloatField(null=True, blank=True)
    latency = models.CharField(max_length=10, null=True, blank=True)
    max_constriction = models.FloatField(null=True, blank=True)
    seventyFivePercentRecovery = models.CharField(max_length=10, null=True, blank=True)
    adv = models.FloatField(null=True, blank=True)
    acv = models.FloatField(null=True, blank=True)
    
    def __str__(self):
        return f"TestResult - {self.id} - {self.timestamp}"
