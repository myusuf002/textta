from django.contrib import admin

from .models import Question
class QuestionAdmin(admin.ModelAdmin):
    list_display = ("category", 'detail', 'label', 'active')
    list_filter = ("category", "active",)
    search_fields = ['category', 'detail']
  
admin.site.register(Question, QuestionAdmin)

from .models import Vectorizer
class VectorizerAdmin(admin.ModelAdmin):
    list_display = ('name', "category", 'created_on')
    list_filter = ("category",)
    search_fields = ['name', 'category']
  
admin.site.register(Vectorizer, VectorizerAdmin)

from .models import Classifier
class ClassifierAdmin(admin.ModelAdmin):
    list_display = ('name', "category", 'created_on')
    list_filter = ("category",)
    search_fields = ['name', 'category']
  
admin.site.register(Classifier, ClassifierAdmin)