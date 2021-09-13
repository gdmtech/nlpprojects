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
# from spacy.lemmatizer import Lemmatizer

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

import string

# for visualisating npl results
from matplotlib import pyplot as plt

import matplotlib.colors as mcolors

# for NER component training
import random
from spacy.util import minibatch, compounding
from pathlib import Path

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
    # doc_list = [doc1, doc2, doc3, doc4, doc5]

    doc = nlp(text)
    print('components=', nlp.pipe_names)
    for ent in doc.ents:
        print(ent.text, ent.label_)

    print('2. Extract the NLP component form the NLLP Model with get_pipe')
    ner = nlp.get_pipe("ner")
    # training data
    TRAIN_DATA = [
        ("Walmart is a leading e-commerce company",
         {"entities": [(0, 7, "ORG")]}),
        ("I reached Chennai yesterday.", {
            "entities": [(19, 28, "GPE")]}),
        ("I recently ordered a book from Amazon",
         {"entities": [(24, 32, "ORG")]}),
        ("I was driving a BMW", {"entities": [(16, 19, "PRODUCT")]}),
        ("I ordered this from ShopClues",
         {"entities": [(20, 29, "ORG")]}),
        ("Fridge can be ordered in Amazon ",
         {"entities": [(0, 6, "PRODUCT")]}),
        ("I bought a new Washer", {"entities": [(16, 22, "PRODUCT")]}),
        ("I bought a old table", {"entities": [(16, 21, "PRODUCT")]}),
        ("I bought a fancy dress", {"entities": [(18, 23, "PRODUCT")]}),
        ("I rented a camera", {"entities": [(12, 18, "PRODUCT")]}),
        ("I rented a tent for our trip", {
            "entities": [(12, 16, "PRODUCT")]}),
        ("I rented a screwdriver from our neighbour",
         {"entities": [(12, 22, "PRODUCT")]}),
        ("I repaired my computer", {"entities": [(15, 23, "PRODUCT")]}),
        ("I got my clock fixed", {"entities": [(16, 21, "PRODUCT")]}),
        ("I got my truck fixed", {"entities": [(16, 21, "PRODUCT")]}),
        ("Flipkart started it's journey from zero",
         {"entities": [(0, 8, "ORG")]}),
        ("I recently ordered from Max", {"entities": [(24, 27, "ORG")]}),
        ("Flipkart is recognized as leader in market",
         {"entities": [(0, 8, "ORG")]}),
        ("I recently ordered from Swiggy",
         {"entities": [(24, 29, "ORG")]})]

    print('3. TRAINING DATA', TRAIN_DATA)
    print('4. ADD NEW LABELS IN THE NER PIPELINE')
    # Adding labels to the `ner`

    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            print('Adding label', ent[2])
            ner.add_label(ent[2])

    print('5. Turn off the pipeline components we dont need')
    # Disable pipeline components you dont need to change
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    # Generate an array of pipe components that are not in exceptions
    unaffected_pipes = [
        pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    print('5.1 Pipes to ignore', unaffected_pipes)

    # print('6. Train in randomised mini-batches')

    # TRAINING THE MODEL
    # Use the * operator to ignore,disable all pipes in the unaffected pipes
    # This routine will not work in Space 3.x and above
    #
    # with nlp.disable_pipes(*unaffected_pipes):

        # Training for 30 iterations
    #    for iteration in range(30):

            # shuffling examples  before every iteration
    #        random.shuffle(TRAIN_DATA)
            #
    #        losses = {}
            # batch up the examples using spaCy's minibatch -
    #        batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
     #       for batch in batches:
    #            texts, annotations = zip(*batch)
     #           nlp.update(
    #                texts,  # batch of texts
     #               annotations,  # batch of annotations
   #               drop=0.5,  # dropout - make it harder to memorise data
     #               losses=losses,
                )
     #       print("Losses", losses)
    print('6. Convert trainig data into .spacy format (docbins)')
    json=json.dumps(TRAIN_DATA)

    # save json file to disk
    # use spacy convert CLI (json converter) to make a doc bin
    print('7. Use the Spacy CLI to generate the models')




print('1. NER UPDATE')
text="India and the UK that previously comprised only a handful of players like IBM or Flipkart in the e-commerce space, is now home to many biggies and giants battling out with each other to reach the top. This is thanks to the overwhelming internet and smartphone penetration coupled with the ever-increasing digital adoption across the country. These new-age innovations not only gave emerging startups a unique platform to deliver seamless shopping experiences but also provided brick and mortar stores with a level-playing field to begin their online journeys without leaving their offline legacies"
ner_update(text)
# - LDA "each document is made up of a distribution of topics and that each topic is in turn made up of a distribution of words.
# The hidden or 'latent' layer - the topic layer" - what is it 'about'
# https://towardsdatascience.com/building-a-topic-modeling-pipeline-with-spacy-and-gensim-c5dc03ffc619
# https://alvinntnu.github.io/NTNU_ENC2045_LECTURES/nlp/topic-modeling-naive.html
# https://www.kaggle.com/thebrownviking20/topic-modelling-with-spacy-and-scikit-learn
