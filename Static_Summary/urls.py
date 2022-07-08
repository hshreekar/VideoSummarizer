from django.urls import path
from . import views

app_name= 'Static_Summary'
urlpatterns = [
    path('',views.main),
]
