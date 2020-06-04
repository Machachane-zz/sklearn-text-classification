# sklearn-text-classification
 
Sklearn text classification using the 20newsgroups dataset 

According to https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html

Algorithm: Multinomial Naive Bayes

Important concepts such as: 
train data, test data, 
TfidfVectorizer, vectorizer, vectors, transform
MultinomialNB, alpha, 
Numpy, asarray,
accuracy, score, filenames, target, shape, 
classifier, fit, prediction, 
metrics, f1-score, average


Relevant actions:

1 - The real data lies in the filenames and target attributes. The target attribute is the integer index of the category

2 - Loading only a sub-selection of the categories by passing the list of the categories to load to the sklearn.datasets.fetch_20newsgroups function

3 - Converting text into vectors of numerical values suitable for statistical analysis

4 - Filtering text for more realistic training 

5 - 