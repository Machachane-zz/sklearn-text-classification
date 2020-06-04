# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 18:32:51 2020

@author: Machachane
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

cats = ['rec.motorcycles', 'rec.sport.baseball', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train', categories=cats)


categories_list = list(newsgroups_train.target_names)
print('\nCategories list:\n', categories_list)

print('\nShape filenames:\n',newsgroups_train.filenames.shape)
print('\nShape target:\n',newsgroups_train.target.shape)
print('\nTarget:\n',newsgroups_train.target[:10])

print('\nConverting text to vectors --------------------------------------------------------\n')

#Converting text to vectors

categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)
print('\nVectors shape:\n', vectors.shape)
print('\nVectors nonzero:\n', vectors.nnz/float(vectors.shape[0]))


print('\nFiltering text for more realistic training ----------------------------------------\n')

#Filtering text for more realistic training 

newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

vectors_test = vectorizer.transform(newsgroups_test.data)
clf = MultinomialNB(alpha=.01)
clf.fit(vectors, newsgroups_train.target)
pred = clf.predict(vectors_test)
metrics.f1_score(newsgroups_test.target, pred, average='macro')

print('\nClf:\n', clf)
print('\nClf fit', clf.fit(vectors, newsgroups_train.target))
print('\nPred:\n', pred)
print('\nF1score:\n', metrics.f1_score(newsgroups_test.target, pred, average='macro'))

print('\n-----------------------------------------------------------------------------------\n')

import numpy as np

def show_top10(classifier, vectorizer, categories):
    feature_names = np.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
        top10 = np.argsort(classifier.coef_[i])[-10:]
        print("%s: %s" % (category, " ".join(feature_names[top10])))
        
        
print('\nShow top10:\n', show_top10(clf, vectorizer, newsgroups_train.target_names))

print('\n-----------------------------------------------------------------------------------\n')

newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=categories)

vectors_test = vectorizer.transform(newsgroups_test.data)
pred = clf.predict(vectors_test)
metrics.f1_score(pred, newsgroups_test.target, average='macro')

print('\nVectors test:\n', vectors_test)
print('\nPred:\n', pred)
print('\nMetrics f1score:\n', metrics.f1_score(pred, newsgroups_test.target, average='macro'))

print('\n-----------------------------------------------------------------------------------\n')


newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=categories)

vectors = vectorizer.fit_transform(newsgroups_train.data)
clf = MultinomialNB(alpha=.01)
clf.fit(vectors, newsgroups_train.target)
vectors_test = vectorizer.transform(newsgroups_test.data)
pred = clf.predict(vectors_test)
metrics.f1_score(newsgroups_test.target, pred, average='macro')

print('\nVectors test:\n', vectors_test)
print('\nPred:\n', pred)
print('\nMetrics f1score:\n', metrics.f1_score(newsgroups_test.target, pred, average='macro'))

