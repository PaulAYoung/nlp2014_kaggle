
# coding: utf-8

# In[5]:

import re
import random
from os import path

import nltk


# In[6]:

class TermScoreClassiffier(nltk.classify.ClassifierI):
    """
    Tries to classify text using scored terms. 
    """
    
    def __init__(self, samples=None, scorer=None, terms=None, key="TermScore", name="TermScoreClassifier"):
        """
        Params:
        
        samples -- a list of samples where each entry is a tuple in format (category,text)
                this argument only works if scorer is also passed. 
                
        scorer -- a function that takes the list of samples and scores them. Must return a dictionary
                in the same format as terms
        
        terms -- a dictionary of terms where keys are the terms and values are dictionaries 
        with the score for each category. ie: {"term": {"c1":0, "c2":10}
        
        key -- The key to used in the returned dictionary. 
        """
        self.key = key
        self.__name__ = name
        
        self.scorer = scorer
        
        if samples and self.scorer:
            self.train(samples)
        else:
            self.terms = terms
        
    def train(self, samples):
        self.terms = self.scorer(samples)
    
    def __call__(self, text):
        """
        Picks a category for text using the term list
        """
        
        if not self.terms:
            raise ValueError("You must train the scorer")
        
        tokens = nltk.word_tokenize(text)
        scores = {}
        
        for c in self.terms.values()[0].keys():
            del self.terms
            scores[c]=0
        
        for w in tokens:
            if w in self.terms:
                for c,s in self.terms[w].iteritems():
                    scores[c] += s
        
        
        totals = scores.items()
        totals.sort(key= lambda s:s[1], reverse=True)
        
        return {self.key: totals[0][0]}


# In[7]:

class TermScoreBagger(TermScoreClassiffier):
    """
    Returns scores for each category based on training data
    """
    
    def __call__(self, text):
        """
        Picks a category for text using the term list
        """

        tokens = nltk.word_tokenize(text)
        scores = {}
        for w in tokens:
            if w in self.terms:
                for c,s in self.terms[w].iteritems():
                    if c in scores:
                        scores[c] += s
                    else:
                        scores[c] = s

        return scores

