
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.conf import settings
from .forms import pipelineForm
from .pipelines import *
import pandas as pd

# Create your views here.

@login_required
def viewIndex(request):     
    form = pipelineForm(request.POST or None, request.FILES or None)
    if request.method == 'POST':
        if form.is_valid():
            data = form.cleaned_data
            document = load_dataset(data['dataset'].file)
            document = preprocessing(document, remove_stopwords=data['stopwords'])
            
            vectorizer, features = feature_extraction(document, ngram=eval(data['n_gram']))
            X = features.values
            y = document['class'].values
            
            X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25)
            print('train set:', X_train.shape, y_train.shape)
            print('test set:', X_test.shape, y_test.shape)

            X_train, y_train = resampling(X_train, y_train, sample=data['resample'])

            models =  {'mnb': GridSearchCV(MultinomialNB(), {'alpha': [1.0, 0.75, 0.5, 0.25,  0.1, 0.01], 'fit_prior': [True, False]}, cv=2),
                       'knn': GridSearchCV(KNeighborsClassifier(), {'n_neighbors': range(1, 20)}, cv=2), 
                       'svc': GridSearchCV(SVC(), {'kernel': ['linear', 'rbf', 'sigmoid', 'poly'], 'gamma': ['auto', 'scale']}, cv=2)} 

            model = models['svc']
            model.fit(X_train, y_train)

            # print("\nbest parameter:", model.best_params_)
            res = pd.DataFrame(model.cv_results_)
            print(res[['params', 'mean_test_score', 'rank_test_score']].sort_values(by='rank_test_score').head())

            evaluate_model(model, X_test, y_test, label=np.unique(y_train))

            # pickle.dump(vectorizer, open('tfidf_enthusiasm_vec_50.pkl', 'wb'))
            # pickle.dump(model, open('svm_enthusiasm_model_50.pkl', 'wb'))
            
    context = {
        'title': "Learn Models",
        'page': "learn",
        'form': form
    }
    return render(request, 'learn/index.html', context)