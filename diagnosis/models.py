from django.db import models


class ImageModel(models.Model):
    image = models.FileField(upload_to='dicoms')
