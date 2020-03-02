from django.urls import path

from . import views

urlpatterns = [
    path('', views.viewIndex, name ="classifier"),
    path('record/save', views.saveRecord, name ="save_record"),
    path('summary', views.viewSummary, name ="summary"),
    ]