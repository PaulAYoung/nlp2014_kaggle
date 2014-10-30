# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import math
import cPickle as pickle
import nltk

from testing_util import sample_sets

# <codecell>

def get_idfs(samples, termer):
    sample_terms = [list(set(termer(s[1]))) for s in samples]
    tdf = nltk.FreqDist(t for l in sample_terms for t in l)
    return tdf

# <codecell>

def get_tfidfs(samples, termer, categories):
    idf = get_idfs(samples, termer)
    
    cat_tfidfs={}
    for i in categories:
        tfidfs={}
        fd = nltk.FreqDist([t for s in sample_sets for t in termer(s[1]) if s[0]==str(i)])

        for t in fd.keys():
            tfidfs[t]=fd.freq(t)/(idf[t] +1)

        cat_tfidfs[i]=tfidfs
    
    return cat_tfidfs

# <codecell>

def get_cat_idfs(samples, termer, categories):
    out = {}
    
    for c in categories:
        cat = []
        
        cat_idf = tfidf_class.get_idfs([s for s in samples if s[0]==c], termer)
        other_idf = tfidf_class.get_idfs([s for s in samples if s[0]!=c], termer)
        
        for t in cat_idf.keys():
            score = cat_idf.freq(t)-other_idf.freq(t)
            cat.append((t, score, cat_idf.freq(t), other_idf.freq(t)))
            cat.sort(key= lambda e: e[1], reverse=True)
            
        out[c]=cat
            
    return out

# <codecell>

def get_cat_idfs(samples, termer, categories):
    out = {}
    
    for c in categories:
        cat = {}
        
        cat_idf = get_idfs([s for s in samples if s[0]==c], termer)
        other_idf = get_idfs([s for s in samples if s[0]!=c], termer)
        
        for t in cat_idf.keys():
            score = cat_idf.freq(t)-other_idf.freq(t)
            cat[t] = (t, score, cat_idf.freq(t), other_idf.freq(t))
            
        out[c]=cat
            
    return out

# <codecell>

default_categories = [str(x) for x in range(1,8)]

class TfidfRater(object):
    
    def __init__(self, termer, categories=default_categories, nterms=200):
        self.termer = termer
        self.categories = categories
        self.nterms=nterms
        
    def train(self, samples):
        cat_tfidfs = get_tfidfs(samples, self.termer, self.categories)
        self.helpful = set()
        for i in self.categories:
            cterms = cat_tfidfs[i].items()
            cterms.sort(key = lambda v: v[1], reverse=True)
            self.helpful.update([w[0] for w in cterms[0:self.nterms]])
    
    def __call__(self, text):
        terms = self.termer(text)
        results = {}
        for t in self.helpful:
            results["has({})".format(t)] = 1 if t in terms else 0

        return results

# <codecell>

class CatIDFRater(TfidfRater):
    
    def train(self, samples):
        cat_idfs = get_cat_idfs(samples, self.termer, self.categories)
        self.helpful = set()
        for i in self.categories:
            cterms = cat_idfs[i].items()
            cterms.sort(key = lambda v: v[1], reverse=True)
            self.helpful.update([w[0] for w in cterms[0:self.nterms]])

# <codecell>

class CatIDFScorer(TfidfRater):
    
    def train(self, samples):
        self.cat_idfs = get_cat_idfs(samples, self.termer, self.categories)
    
    def __call__(self, text):
        terms = self.termer(text)
        results = {"score({})".format(c):0 for c in self.categories}
        for t in terms:
            for c in self.categories:
                if t in self.cat_idfs[c]:
                    results["score({})".format(c)] += self.cat_idfs[c][t][1]
                else:
                    others = [cat[t][1] for cat in self.cat_idfs if t in cat]
                    results["score({})".format(c)] += -(sum(others)/(len(others)+1))

        return results

