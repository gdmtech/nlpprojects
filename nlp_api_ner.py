# NLP test API
# David Williams
# Aug 1, 2021
# using Spacy - spaCy is an open-source software library for
# advanced natural language processing, written in the
# programming languages Python and Cython.

# explosion AI spacy 3.0 contains many new features

# Depedencies
# ml_datasets

import pprint

from ml_datasets import imdb
import spacy
# import spefiic spacy components that we will use in a pipeline

# I had the same issue. So it turns out spacy.lemmatizer is not available in spacy v3. You need to use spacy v2. lemmanizer is now a defualt component
#from spacy.lemmatizer import Lemmatizer

# import the list of stopwords for the EN language
from spacy.lang.en.stop_words import STOP_WORDS

# count frequency
from collections import Counter
# most commmon punctuation in English
from string import punctuation
# we need pandas to manipulate text input
import pandas as pd

# import a progress bar - https://tqdm.github.io/
from tqdm.auto import tqdm

# DocBin is spacys new way to store Docs in a
# binary format for training later
from spacy.tokens import DocBin

from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE


# lets start learning ANOTHER NLP toolit (text focused)
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# we need GENSIM for topic modelling functions
import gensim
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.test.utils import datapath

import string

# for visualisating npl results
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors


# endpoint TEXT SUMMARY
# summarise using sentences with high frequency words (normalised)

# load the spacy language model into memory (the model must be installed by pip as its large)
# the nlp contains parsing pipelines - check with v
print('load the spacy language model into memory (the model must be installed by pip)')
nlp = spacy.load("en_core_web_lg")
print('pipelines', nlp.pipe_names)
print('en_core_web_lg installed')


# END POINT MOJO TEXT SUMMARY

def ner_update(text):

    # Update the NER trainer

    # compile documents into one structure
    #doc_list = [doc1, doc2, doc3, doc4, doc5]

    doc = nlp(text)
    print('components=', nlp.pipe_names)
    for ent in doc.ents:
        print(ent.text, ent.label_)

    # create a training data set with new NER required
    # disable the other pipes


print('1. NER UPDATE')
text = "India and the UK that previously comprised only a handful of players like IBM or Flipkart in the e-commerce space, is now home to many biggies and giants battling out with each other to reach the top. This is thanks to the overwhelming internet and smartphone penetration coupled with the ever-increasing digital adoption across the country. These new-age innovations not only gave emerging startups a unique platform to deliver seamless shopping experiences but also provided brick and mortar stores with a level-playing field to begin their online journeys without leaving their offline legacies"
ner_update(text)
# - LDA "each document is made up of a distribution of topics and that each topic is in turn made up of a distribution of words.
# The hidden or 'latent' layer - the topic layer" - what is it 'about'
# https://towardsdatascience.com/building-a-topic-modeling-pipeline-with-spacy-and-gensim-c5dc03ffc619
# https://alvinntnu.github.io/NTNU_ENC2045_LECTURES/nlp/topic-modeling-naive.html
# https://www.kaggle.com/thebrownviking20/topic-modelling-with-spacy-and-scikit-learn
