from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('api/recommend/', views.recommend_laptops, name='recommend_laptops'),
] 