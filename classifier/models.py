from django.db import models
from ckeditor.fields import RichTextField


# Create your models here.
class Question(models.Model):
    category = models.CharField(max_length=16, blank=True, null=True)    
    detail = models.TextField(max_length=1024, blank=True, null=True)    
    label = models.TextField(max_length=1024, blank=True, null=True)
    active = models.BooleanField(blank=True, null=True, default=True)
    
    def __str__(self): return self.category

    class Meta:
        verbose_name_plural = "Questions"   

class Vectorizer(models.Model):
    name = models.CharField(max_length=512, blank=True, null=True)
    category = models.ForeignKey(Question, on_delete=models.CASCADE)
    vector = models.FileField(upload_to='vector/')
    created_on = models.DateField(auto_now=True)
    detail = RichTextField(blank=True, null=True)

    def __str__(self): return self.name

    class Meta:
        verbose_name_plural = "Vectorizers"   

class Classifier(models.Model):
    name = models.CharField(max_length=512, blank=True, null=True)
    category = models.ForeignKey(Question, on_delete=models.CASCADE)
    model = models.FileField(upload_to='models/')
    created_on = models.DateField(auto_now=True)
    detail = RichTextField(blank=True, null=True)

    def __str__(self): return self.name

    class Meta:
        verbose_name_plural = "Classifiers"   