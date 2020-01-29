
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.conf import settings
from .forms import pipelineForm
from .pipelines import *
import pandas as pd
import glob
import os

# Create your views here.

@login_required
def viewIndex(request):
    for filename in glob.glob("static/"+request.user.username+"*.pkl"):        
        os.remove(os.path.join(filename))
    
    form = pipelineForm(request.POST or None, request.FILES or None)
    if request.method == 'POST':
        if form.is_valid():
            return learnModel(request, form.cleaned_data)
            
    context = {
        'title': "Learn Models",
        'page': "learn",
        'form': form
    }
    return render(request, 'learn/index.html', context)

@login_required
def learnModel(request, data):
    document = load_dataset(data['dataset'].file)
    document = preprocessing(document, remove_stopwords=data['stopwords'])
    


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
        'dataset': data['dataset'].name,
        'category': data['category'],
        'question': data['question'],
        'shape': document['class'].shape,
        'distribution': document['class'].value_counts().to_dict(),
        'remove_stopwords': data['stopwords'],
        'train_set': y_train.size,
        'test_set':  y_test.size,
        'features': X_train.shape[1],
        'ngram': ngram,
        'resample': data['resample'],
        'eval_models': eval_models
    }
    
    pickle.dump(vectorizer, open('static/'+request.user.username+'_tfidf_'+data['category'].lower()+'_vec.pkl', 'wb'))
    pickle.dump(models['svc'], open('static/'+request.user.username+'_svm_'+data['category'].lower()+'_model.pkl', 'wb'))
    pickle.dump(models['mnb'], open('static/'+request.user.username+'_mnb_'+data['category'].lower()+'_model.pkl', 'wb'))
    pickle.dump(models['knn'], open('static/'+request.user.username+'_knn_'+data['category'].lower()+'_model.pkl', 'wb'))
    
    return render(request, 'learn/learn.html', context)

@login_required
def viewVectorizer(request):
    context = {
        'title': "Vectorizer",
        'page': "learn"
    }
    return render(request, 'learn/vectorizer.html', context)
