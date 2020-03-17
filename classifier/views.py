import os
import json
import string
import pickle
import pandas as pd
from time import sleep

from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.conf import settings

from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from django.http import HttpResponse, JsonResponse
from django import forms

from .models import Question, Vectorizer, Classifier
from .audio import convert_audio, info_audio
from .transcribe import upload_to_aws, remove_from_aws, transcribe_aws

class QuestionChoiceField(forms.ModelChoiceField):
    def label_from_instance(self, obj): return obj.detail

class predictForm(forms.Form):
    # name = forms.CharField(label='Name', max_length=512)
    question = QuestionChoiceField(queryset=Question.objects.all())
    answer = forms.CharField(label='Answer', max_length=512, widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 3}))

    # for bootstrap styling
    def __init__(self, *args, **kwargs):
        super(predictForm, self).__init__(*args, **kwargs)
        for visible in self.visible_fields():
            if hasattr(visible.field.widget, 'input_type'):
                if visible.field.widget.input_type in ['radio', 'checkbox']:
                    visible.field.widget.attrs['class'] = 'form-check-input'
                else:
                    visible.field.widget.attrs['class'] = 'form-control'

# Create your views here.

def viewIndex(request):  
    
    form = predictForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            return viewPredict(request, form.cleaned_data)
            
    context = {
        'title': "Select Question",
        'page': "classifier",
        'form': form
    }
    
    return render(request, 'classifier/index.html', context)

def saveRecord(request):
    request.session['recorded'] = 1
    print("... save record ...")
    audio_data = request.FILES['audio']
    audio_name = request.POST['name']
    default_storage.save('audio/'+audio_name, ContentFile(audio_data.read()))
    info_audio(settings.MEDIA_ROOT+'/audio/'+audio_name)
    print()
    # Preprocessing
    print("... preprocessing ...")
    convert_audio(settings.MEDIA_ROOT+'/audio/'+audio_name, audio_name)
    info_audio(audio_name)
    print()

    # uploading 
    print("... uploading ...")
    upload_to_aws(audio_name, audio_name)
    
    # Transcribing 
    print("... transcribing ...")
    text = transcribe_aws(audio_name)
    
    print()
    print("... clean record ...")
    
    remove_from_aws(audio_name)
    default_storage.delete('audio/'+audio_name)
    os.remove(audio_name)
    
    data = {
        'name': request.POST['name'],
        'transcribe': text,
        'status': "saved"
    }
    return JsonResponse(data)

def viewPredict(request, response):  
    with open(os.path.join(settings.STATIC_ROOT, 'classifier/stopwords.txt'), 'rb') as f:
                stopwords = f.read().splitlines()
    
    # preprocessing answer
    raw = pd.Series([response['answer']]) 
    raw = raw.apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
    raw = raw.str.replace('[{}]'.format(string.punctuation), '')
    raw = raw.str.lower()

    print(response['question'])
    # load vectorizer (from corpus of question)
    vector = Vectorizer.objects.get(category=response['question'])    
    vectorizer = pickle.load(open(vector.vector.path, 'rb'))

    # vectorize answer using tf-idf
    raw_vectors = vectorizer.transform(raw)
    feature_names = vectorizer.get_feature_names()
    dense = raw_vectors.todense()
    denselist = dense.tolist()
    
    models = Classifier.objects.filter(category=response['question'])

    predicts = []
    question = Question.objects.filter(category=response['question']).first()
    labels = json.loads(question.label)

    for classifier in models: 
        model = pickle.load(open(classifier.model.path, 'rb'))
        predict = model.predict(denselist)
        predicts.append({'nilai': predict[0],
                         'label': labels[str(predict[0])],
                         'model': classifier.name})
        

    context = {
        'title': "Classified Answer",
        'page': "classifier",
        'response': response,
        'predicts': predicts
    }
    return render(request, 'classifier/predict.html', context)

@login_required
def viewSummary(request):     
    context = {
        'title': "Summary Classifier",
        'page': "classifier",
        'questions': Question.objects.all(),
        'vectorizers': Vectorizer.objects.all(),
        'classifiers': Classifier.objects.all()
    }
    return render(request, 'classifier/summary.html', context)