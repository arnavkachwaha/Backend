from django. db import models

class Eyes (models.Model):
    videofile = models.FileField(upload_to='./') 