from django.urls import path

from . import views

urlpatterns = [
    path('', views.viewIndex, name ="classifier"),
    path('summary', views.viewSummary, name ="summary"),
    ]