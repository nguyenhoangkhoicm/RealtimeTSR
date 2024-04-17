
from django.urls import path

from web import views

urlpatterns = [
    path('', views.index, name='index'),
    path('identification/', views.Identification, name='Identification'),
]


