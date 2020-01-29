from django.urls import path

from . import views

urlpatterns = [
    path('', views.viewIndex, name ="learn"),
    path('vectorizer', views.viewVectorizer, name ="vectorizer"),
    ]