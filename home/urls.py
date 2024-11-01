from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path('video_feed/',views.video_feed,name='video-feed'),
    path('height/',views.video_feed_2,name='video-feed-2'),
    path('', views.home,name='home'),
]