from django.conf import settings

import os
import string
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import pickle 
# from wordcloud import WordCloud

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score


with open(os.path.join(settings.STATIC_ROOT, 'classifier/stopwords.txt'), 'rb') as f:
    STOPWORDS = f.read().splitlines()

def load_dataset(filename):
    print("load dataset")
    documents = pd.read_csv(filename)
    print(" > shape:",documents['class'].shape)
    print(" > distribution:",documents['class'].value_counts().to_dict())

    # print("review dataset")
    # print(documents.head(4))

    return documents

def preprocessing(documents, remove_stopwords=False, encode=False):
    print("preprocessing")
    if remove_stopwords:
        print(" > remove stopwords")
        documents['script'] = documents['script'].apply(lambda x: ' '.join([word for word in x.split() if word not in (STOPWORDS)]))

    if encode:
        print(" > encode label")
        Encoder = LabelEncoder()
        documents['class'] = Encoder.fit_transform(documents['class'])

    print(" > remove punctuation")
    documents['script'] = documents['script'].str.replace('[{}]'.format(string.punctuation), '')

    print(" > lowercasing")
    documents['script'] = documents['script'].str.lower()

    return documents

def feature_extraction(documents, ngram=(1, 2)):
    print("extract features with ngram", ngram)
    vectorizer = TfidfVectorizer(ngram_range=ngram)
    vectors = vectorizer.fit_transform(documents['script'])
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    features = pd.DataFrame(denselist, columns=feature_names)
    print(" > features size:", len(feature_names))
    return vectorizer, features

def resampling(X_train, y_train, sample='minority'):
    if sample != 'None':
        print("\nresampling with", sample)
        smote = SMOTE(sample, k_neighbors=2)

        X_train, y_train =  smote.fit_sample(X_train, y_train)
        print(' > train set:', X_train.shape, y_train.shape)
        print(' > distribution:', np.unique(y_train, return_counts=True))
        print()
    return X_train, y_train
    

def evaluate_model(model, X_test, y_test, label=[1, 2, 3]):
    print("\ntest model")
    y_pred = model.predict(X_test)
    print("true class    :", y_test)
    print("predict class :", y_pred)
    print()

    df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred, labels=label), index = label, columns = label)
    sn.heatmap(df_cm, annot=True, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()
    print('\nevaluate model')
    print(' > accuracy: %.5f' % accuracy_score(y_test, y_pred))
    print(' > f1-score: %.5f' % f1_score(y_test, y_pred, average='macro'))
    print(' > recall: %.5f' % recall_score(y_test, y_pred, average='macro'))
    print(' > precision: %.5f' % precision_score(y_test, y_pred, average='macro'))
    print()
    # print('\nclasification report:\n', classification_report(y_test, y_pred))
