# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import re
import random
from os import path

import nltk

import term_scoring

# <codecell>

path_train = path.join(path.curdir, "train.txt")
path_final_testing = path.join(path.curdir, "test.csv")

# <codecell>

def file_to_sets(fname, ignore_header=True):
    """
    Takes a file where each line is in the format "category,text" and turns it into a list of tuples
    in format "(category, text)"
    """
    
    f = open(fname, 'r')
    
    if ignore_header:
        # This skips the first line of the file
        next(f)
    
    out = []
    for line in f:
        # iterate over lines, use simple regex to separate the category from text
        out.append(re.match(r"(\d+),(.+$)", line).groups())
    
    f.close()
    return out

# <codecell>

sample_sets = file_to_sets(path_train, ignore_header=False)
final_sets = file_to_sets(path_final_testing, ignore_header=True)

# <codecell>

def get_sets(samples, test_fraction=3):
    """
    takes a set of samples, shuffles them, then returns two lists, train_sets and test_sets. 
    The size of test_sets is len(samples)/test_fraction, train_sets is the remainder. 
    """
    
    # don't shuffle the sample list as that will affect the list passed in
    l = samples[:]
    random.shuffle(l)
    
    test_size = int(len(l)/test_fraction)
    test_sets = l[0:test_size]
    train_sets = l[test_size:]
    
    return train_sets, test_sets

# <codecell>

def get_folds(samples, folds=3):
    """
    Returns of list of folds from samples
    """
    
    # don't shuffle the sample list as that will affect the list passed in
    l = samples[:]
    random.shuffle(l)
    out = []
    chunk_size = int(len(samples)/folds)
    sections = range(0, len(samples)+1, chunk_size)
    sections[-1]= None
    
    for i in range(0,len(sections)-1):
        out.append(l[sections[i]:sections[i+1]])
    
    return out

# <codecell>

stopwords = nltk.corpus.stopwords.words('english')

def get_terms(t):
    tokens = nltk.word_tokenize(t)
    return [w for w in tokens if w not in stopwords]

# <codecell>

def create_training_sets (feature_function, items):
    # Create the features sets.  Call the function that was passed in.
    # For names, key is the name, and value is the gender
    featuresets = [(feature_function(key), value) for (key, value) in items]
    
    # Divided training and testing in half.  Could divide in other proportions instead.
    halfsize = int(float(len(featuresets)) / 2.0)
    train_set, test_set = featuresets[halfsize:], featuresets[:halfsize]
    return train_set, test_set

# <codecell>

def make_classifier(feature_extractor, train, classifier=nltk.classify.NaiveBayesClassifier):
    """
    creates a classifier based on the feature_extractor, trains it with train and
    tests it with test
    """
    train_features = [(feature_extractor(text), category) for (category, text) in train]
    
    cl = classifier.train(train_features)
    return cl

# <codecell>

def fold_test_extractor(feature_extractor, samples, folds=3, classifier=nltk.NaiveBayesClassifier.train, confusion=False, **kwargs):
    """
    Tests a feature extractor with a set of sample tuples in format (category, text)

    Params:
    feature_extractor -- The feature extractor function to use
    samples -- the samples to test with
    folds -- the number of folds to use in testing
    classifier--classification method
    """
    
    folds = get_folds(samples, folds)
    
    for i in range(0, len(folds)):
        train_sets = [f for idx, s in enumerate(folds) for f in s if idx !=i]
        
        if isinstance(feature_extractor, FeatureExtractor):
            feature_extractor.train_extractors(train_sets)
        
        if isinstance(feature_extractor, term_scoring.TermScoreBagger):
            feature_extractor.train(train_sets)
        
        train = [(feature_extractor(text), category) for category, text in train_sets]
        test = [(feature_extractor(text), category) for category, text in  folds[i]]
        
        if type(feature_extractor) is FeatureExtractor:
            cl = feature_extractor.get_classifier(samples, classifier, retrain=False)
        else:
            cl = make_classifier(feature_extractor, samples, classifier)
            
        print "test {} - {:.3%}".format(i, nltk.classify.accuracy(cl, test))
        
        if confusion:
            results = [cl.classify(i[0]) for i in test]
            reference = [i[1] for i in test]
            cm = nltk.ConfusionMatrix(reference, results)
            print(cm.pp(sort_by_count=True, show_percents=True, truncate=9))

# <codecell>

def make_submission(classifier, samples, writeto=None):
    out = []
    for s in samples:
        out.append((s[1], classifier.classify(s[0])))
    
    if writeto:
        out_file = open(writeto, 'w')
        out_file.write("Id,Category\n")
        for n, c in out:
            out_file.write("{},{}\n".format(n,c))
    
    return out

# <codecell>

def make_feature(extractor, samples):
    return [(extractor(text), category) for category, text in samples]

# <codecell>

class WordSearcher(object):
    def __init__(self, wordlist=None, key="", name="wordsearcher"):
        self.wordlist = wordlist if wordlist else []
        self.key = key
        self.name=name
    
    def add(self, wordlist):
        self.wordlist.extend(wordlist)
    
    def clear(self):
        self.wordlist = []
    
    def __call__(self, text):
        out = {}
        for word in self.wordlist:
            out["{}has({})".format(self.key, word)] = 1 if re.search(word, text, re.IGNORECASE) else 0
        return out

# <codecell>

class FeatureExtractor(object):
    """A class to make it easy to combine and shuffle around feature extractors"""
    
    def __init__(self, extractors=None, name="ExtractorCollection"):
        """
        Takes a list of extractors to use in extracting features. 
        Extractors should take a piece of text and return a dictionary where the key is
        the desired key and the value is the feature value. 
        """
        if extractors is None:
            extractors = []
            
        if type(extractors) is not list:
            extractors = [extractors]
        
        self.__name__ = name
        self.extractors = extractors
        self.trained_extractors = []
        
    def __call__(self, text):
        features = {}
        for e in self.extractors:
            f = e(text)
            for k, v in f.iteritems():
                features[k]=v
        
        for e in self.trained_extractors:
            f = e(text)
            for k, v in f.iteritems():
                features[k]=v
        
        return features
    
    def add_extractor(self,extractor, trained=False):
        if trained==True:
            self.trained_extractors.append(extractor)
        else:
            self.extractors.append(extractor)
    
    def train_extractors(self, samples):
        for t in self.trained_extractors:
            t.train(samples)
    
    def test_extractors(self, samples, folds=3, classifier=nltk.NaiveBayesClassifier):
        """
        Runs a test of each individual extractor using samples to train and test. 
        """
        
        for e in self.extractors:
            print "Extractor: {}".format(e.__name__)
            fold_test_extractor(e, samples, folds, classifier=classifier)
        
        for e in self.trained_extractors:
            print "Extractor: {}".format(e.__name__)
            fold_test_extractor(e, samples, folds, classifier=classifier)
    
    def test(self, samples, folds=3, classifier=nltk.NaiveBayesClassifier.train, num_tests=1, **kwargs):
        for i in range(0, num_tests):
            print "**************************"
            print "Run {}".format(i)
            print "**************************"
            fold_test_extractor(self, samples, folds, classifier, **kwargs)
    
    def get_classifier(self, samples, classifier=nltk.classify.NaiveBayesClassifier, retrain=True):
        """
        Creates a classifier based on the extractor, using samples to train and the classifier as the
        method.
        """
        if retrain:
            self.train_extractors(samples)
        return make_classifier(self, samples, classifier)
    
    def show_errors(self, samples, classifier=nltk.classify.NaiveBayesClassifier, folds=10):
        folds = get_folds(samples, folds=folds)
        train_sets = [f for s in folds[1:] for f in s]
        test_sets = folds[0]
        
        self.train_extractors(train_sets)
        
        cl = self.get_classifier(train_sets, classifier=classifier, retrain=False)
        
        errors = []
        for tag, text in test_sets:
            guess = cl.classify(self(text))
            if guess != tag:
                errors.append((guess, tag, text))
                
        return errors

