# Generated by Django 2.1.2 on 2021-12-11 17:08

import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Static_Summary', '0002_auto_20211211_2123'),
    ]

    operations = [
        migrations.AlterField(
            model_name='document',
            name='videoFile',
            field=models.FileField(default='dummy.txt', upload_to='documents/', validators=[django.core.validators.FileExtensionValidator(['mpg'])]),
        ),
    ]
