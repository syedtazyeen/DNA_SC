import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# ML model class
class Model:

    def __init__(self,kmer):
      self.algo = MultinomialNB()
      self.vectorizer = CountVectorizer(analyzer='char', ngram_range=(kmer, kmer))
      self.accuracy = None

    def train(self,x,y):
      self.algo.fit(x,y)

    def test(self, x, y):
      y_pred = self.get_prediction(x)
      self.accuracy = accuracy_score(y, y_pred)

    def get_prediction(self,x):
      return self.algo.predict(x)