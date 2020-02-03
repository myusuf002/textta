
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.core.files import File
from django.conf import settings
from .forms import pipelineForm


import pandas as pd
import numpy as np
import glob
import os

import pickle 
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from .pipelines import load_dataset, preprocessing, feature_extraction, resampling, evaluate_model

from classifier.models import Question, Vectorizer, Classifier
# Create your views here.

@login_required
def viewIndex(request):
    for filename in glob.glob(request.user.username+"*.pkl"):        
        os.remove(os.path.join(filename))
    
    form = pipelineForm(request.POST or None, request.FILES or None)
    if request.method == 'POST':
        if form.is_valid():
            return learnModel(request)
            
    context = {
        'title': "Learn Models",
        'page': "learn",
        'form': form
    }
    return render(request, 'learn/index.html', context)

@login_required
def learnModel(request):
    data = request.POST
    files = request.FILES

    document = load_dataset(files['dataset'].file)
    document = preprocessing(document, remove_stopwords=data['stopwords'] if data.__contains__('stopwords') else False)
    


    if data['n_gram'] == '(1, 1)': ngram = 'Unigram'
    elif data['n_gram'] == '(1, 2)': ngram = 'Unigram - Bigram'
    elif data['n_gram'] == '(2, 2)': ngram = 'Bigram'
    elif data['n_gram'] == '(1, 3)': ngram = 'Unigram - Trigram'
    elif data['n_gram'] == '(2, 3)': ngram = 'Bigram - Trigram'
    elif data['n_gram'] == '(3, 3)': ngram = 'Trigram'
    else: ngram = 'None'

    vectorizer, features = feature_extraction(document, ngram=eval(data['n_gram']))
    X = features.values
    y = document['class'].values
    
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25)
    X_train, y_train = resampling(X_train, y_train, sample=data['resample'])

    models =  {'svc': GridSearchCV(SVC(), {'kernel': ['linear', 'rbf', 'sigmoid', 'poly'], 'gamma': ['auto', 'scale']}, cv=2),
               'mnb': GridSearchCV(MultinomialNB(), {'alpha': [1.0, 0.75, 0.5, 0.25,  0.1, 0.01], 'fit_prior': [True, False]}, cv=2),
               'knn': GridSearchCV(KNeighborsClassifier(), {'n_neighbors': range(1, len(y_train)//2)}, cv=2)} 
    
    eval_models = {}
    for name, model in models.items():        
        model.fit(X_train, y_train)
        res = pd.DataFrame(model.cv_results_)

        res_sort = res[['params', 'mean_test_score', 'rank_test_score']].sort_values(by='rank_test_score').head(3)
        res_disp = res_sort[['params', 'mean_test_score']]
        res_disp['mean_test_score'] = res_disp['mean_test_score'].apply(lambda x:np.round(x, 4))
        y_pred, evaluation, conf_matrix = evaluate_model(model, X_test, y_test, label=np.unique(y_train))

        if name == 'svc': full_name = "Support Vector Machine"
        elif name == 'mnb': full_name = "Multinomial Naive Bayes"
        elif name == 'knn': full_name = "K-Nearest Neighbour"

        res_eval_model = {
            'model_name': full_name,
            'best_params': model.best_params_,
            'y_test': y_test,
            'y_pred': y_pred,
            'evaluation': evaluation,
            'cros_val': res_disp,
            'conf_matrix': conf_matrix
        }
        eval_models[name] = res_eval_model
    

    context = {
        'title': "Result Pipeline",
        'page': "learn",
        'user': request.user.username,
        'dataset': files['dataset'].name,
        'category': data['category'],
        'question': data['question'],
        'shape': document['class'].shape,
        'distribution': document['class'].value_counts().to_dict(),
        'remove_stopwords': data['stopwords'] if data.__contains__('stopwords') else False,
        'train_set': y_train.size,
        'test_set':  y_test.size,
        'features': X_train.shape[1],
        'ngram': ngram,
        'resample': data['resample'],
        'eval_models': eval_models
    }
    
    pickle.dump(vectorizer, open(request.user.username+'_tfidf_'+data['category'].lower()+'_vec.pkl', 'wb'))
    pickle.dump(models['svc'], open(request.user.username+'_svm_'+data['category'].lower()+'_model.pkl', 'wb'))
    pickle.dump(models['mnb'], open(request.user.username+'_mnb_'+data['category'].lower()+'_model.pkl', 'wb'))
    pickle.dump(models['knn'], open(request.user.username+'_knn_'+data['category'].lower()+'_model.pkl', 'wb'))
    
    return render(request, 'learn/learn.html', context)

@login_required
def viewVectorizer(request):
    if request.method == 'POST':        
        context = {
            'title': "Vectorizer",
            'page': "learn",
            'data': request.POST
        }
        
        if Question.objects.filter(category=request.POST['category'].lower()).exists():
            print('Category exist!')
            question = Question.objects.get(category=request.POST['category'].lower())
            question.detail = request.POST['question']
            question.save()

            tfidf_file = open(request.user.username+'_tfidf_'+request.POST['category'].lower()+'_vec.pkl', 'rb')
            vectorizer = Vectorizer.objects.get(category=question)
            vectorizer.vector = File(tfidf_file)
            vectorizer.save()

            svc_file = open(request.user.username+'_svm_'+request.POST['category'].lower()+'_model.pkl', 'rb')
            svc_model = Classifier.objects.get(category=question)
            svc_model.model=File(svc_file)
            svc_model.save()

            mnb_file = open(request.user.username+'_mnb_'+request.POST['category'].lower()+'_model.pkl', 'rb')
            mnb_model = Classifier.objects.get(category=question)
            mnb_model.model=File(mnb_file)
            mnb_model.save()

            knn_file = open(request.user.username+'_knn_'+request.POST['category'].lower()+'_model.pkl', 'rb')
            knn_model = Classifier.objects.get(category=question)
            knn_model.model=File(knn_file)
            knn_model.save()
            

        else:
            question = Question(category=request.POST['category'].lower(), 
                           detail=request.POST['question'], 
                           active=True)
            question.save()

            tfidf_file = open(request.user.username+'_tfidf_'+request.POST['category'].lower()+'_vec.pkl', 'rb')
            vectorizer = Vectorizer(name="tfidf_"+request.POST['category'].lower()+"_vector", 
                                    category=question, 
                                    vector=File(tfidf_file),
                                    detail=request.POST['vector'])
            vectorizer.save()

            svc_file = open(request.user.username+'_svm_'+request.POST['category'].lower()+'_model.pkl', 'rb')
            svc_model = Classifier(name="svm_"+request.POST['category'].lower()+"_model", 
                                    category=question, 
                                    model=File(svc_file),
                                    detail=request.POST['svc_detail'])
            svc_model.save()

            mnb_file = open(request.user.username+'_mnb_'+request.POST['category'].lower()+'_model.pkl', 'rb')
            mnb_model = Classifier(name="mnb_"+request.POST['category'].lower()+"_model", 
                                    category=question, 
                                    model=File(mnb_file),
                                    detail=request.POST['mnb_detail'])
            mnb_model.save()

            knn_file = open(request.user.username+'_knn_'+request.POST['category'].lower()+'_model.pkl', 'rb')
            knn_model = Classifier(name="knn_"+request.POST['category'].lower()+"_model", 
                                    category=question, 
                                    model=File(knn_file),
                                    detail=request.POST['knn_detail'])
            knn_model.save()

        return render(request, 'learn/vectorizer.html', context)
    else: return redirect(viewIndex)
