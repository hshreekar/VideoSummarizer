from django.db import models
from django.core.validators import FileExtensionValidator

def validate_geeks_mail(value):
	print("VALUE",value)
	# a = str(value).split('.')[-1]
	# if a in ['mp4','mkv','mpg']:
	# # 	return
	# # else:
	# 	raise ValueError("This field accepts only mp4,mkv,mpg format")
	# a = str(value) a.split('.')[-1]:

    # else:
    # 	raise ValidationError("This field accepts only mp4,mkv,mpg format")

class Document(models.Model):
	
	
	videoFile = models.FileField(upload_to='documents/',default='dummy.txt',validators =[FileExtensionValidator(['mpg','mp4','mkv'])])

