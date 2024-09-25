from django import forms
from Cyclops.models import Eyes


class VideoForm(forms.ModelForm):
    class Meta:
        model= Eyes
        fields= ["videofile"]
