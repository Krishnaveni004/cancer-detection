from .models import *
from django.forms import ModelForm

class FileForm(ModelForm):
    class Meta:
        model = ImageModel
        fields = '__all__'