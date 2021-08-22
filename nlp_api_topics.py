# NLP test API
# David Williams
# Aug 1, 2021
# using Spacy - spaCy is an open-source software library for
# advanced natural language processing, written in the
# programming languages Python and Cython.

# explosion AI spacy 3.0 contains many new features

# Depedencies
# ml_datasets

import pyLDAvis.gensim
import pprint
from wordcloud import WordCloud
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


# endpoint TEXT SUMMARY
# summarise using sentences with high frequency words (normalised)

# load the spacy language model into memory (the model must be installed by pip as its large)
# the nlp contains parsing pipelines - check with v
print('load the spacy language model into memory (the model must be installed by pip)')
nlp = spacy.load("en_core_web_lg")
print('pipelines', nlp.pipe_names)
print('en_core_web_lg installed')


# END POINT MOJO TEXT SUMMARY

def topic_model2():

    # SETUP A NEW STOP WORD LIST THAT IS SPECIFIC TO THE TARGET TEXT
    # My list of stop words.
    # we will use one document
    input = "I wish that I am somewhere else with Mrs. Smith.  I want to fly to new york or tokyo.  Anywhere but here,say.  I'll book a ticket tomorrow.  Do I have any money?  I will need money when I'm booking"
    doc = nlp(input)
    input2 = "He rushed to the train to book the ticket.  But he didn't have any food.  He ran to the shop to buy food.  It was very expensive.  How can he afford it?  He must go to the bank"
    doc2 = nlp(input2)

    stop_list = ["Mrs.", "Ms.", "say", "WASHINGTON", "'s", "Mr.", ]
    # Updates spaCy's default stop words (mapped to STOP_WORDS) list with my additional words.
    nlp.Defaults.stop_words.update(stop_list)
    # Iterates over the words in the stop words list and resets the "is_stop" flag.
    # STOP WORDS are pretrained in a language, e.g ENG
    for word in STOP_WORDS:
        lexeme = nlp.vocab[word]
        lexeme.is_stop = True

    # PROCESSING THE TEXT
    # ALL words in the language Vocab have attributes that are assigned (e.g. is_stop)
    # NLP has already lemmatized; now we need to create a new doc that uses root tokens
    clean_text = lemmatize(doc)
    #print('lemmatized doc',clean_text)
    # remove the stop words and export text
    clean_text = remove_stopwords(clean_text)

    # repeat for dox2
    clean_text2 = lemmatize(doc2)
    clean_text2 = remove_stopwords(clean_text2)
    print("TEXT IS READY FOR TOPIC MODELLING:", clean_text)
    print("TEXT IS READY FOR TOPIC MODELLING:", clean_text2)
    # PREPARE THE DOCUMENT LIST FOR TOPIC MODELLING (GENSIM only accepts TEXT so be careful that you are inputting Tokens.text)
    doc_list = []
    doc_list.append(clean_text)
    doc_list.append(clean_text2)

    # GENSHIM SETUP
    # Creates, which is a mapping of word IDs to words.
   # dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
    words = corpora.Dictionary(doc_list)

    # Turns each document into a bag of words.
    corpus = [words.doc2bow(doc) for doc in doc_list]

    # generate the an unsuperivsed LDA statistic analysis; using 10 topics
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=words,
                                                num_topics=4,
                                                random_state=2,
                                                update_every=1,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)
    # print the model result
    # the model attributes such as topics can be manipulated
    pprint.pprint(lda_model.print_topics(num_words=4))
    top_topics = lda_model.top_topics(corpus)  # , num_words=20)
    pprint.pprint(lda_model.top_topics(corpus))

    # EXPLORE

    # visualise the topic output - show the most likely topic fro each document
    for doc in doc_list:
        print(lda_model.get_document_topics(words.doc2bow(doc)))
    # visualise
    #lda_display = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary, sort_topics=False)
    # pyLDAvis.display(lda_display)

    # save the model to the folder output
    lda_model.save('./output/lda/test')

    # reload the model to lda_model2
    #lda_model2 = LdaModel.load('./output/lda/test')
    # visualise the topic output - show the most likely topic fro each document

    # TEST with unseen data
    test = ["book", "bread", "eat", "train", "ticket"]
    print('test model with unseen data to see its likely topics ', test)
    print(lda_model.get_document_topics(words.doc2bow(test)))
    print('Visualise')
   # pyLDAvis.enable_notebook()
    #vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)


def lemmatize(doc):
    # Made by David
    # Input is a spacy doc
    # Return a new spacy doc with lemmatized tokens
    doc = [token.lemma_ for token in doc]
    # makea string with the lemma list
    doc = u' '.join(doc)
    # convertback to a list of token
    return(nlp.make_doc(doc))


def remove_stopwords(doc):
    # Made by David
    # Input is a spacy doc
    # Remove stopwords and punctuation and spaces.
    # Return a list of valid tokens.text
    removed = []
    for token in doc:
        if not (token.is_stop or token.is_punct or token.is_space):
            removed.append(token.text)
    return(removed)

    # The add_pipe function appends our functions to the DEFAULT pipeline.


def wordcloud(papers):
    # Import the wordcloud library
    # Join the different processed titles together.
    long_string = ','.join(list(papers['paper_text_processed'].values))
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=5000,
                          contour_width=3, contour_color='steelblue')
    # Generate a word cloud
    wordcloud.generate(long_string)
    # Visualize the word cloud
    wordcloud.to_image()


# open the txt file for analysis
print('COGNIAI NLP DEMO')

print('4. Topic Modelling using spacy3 and gensim')
topic_model2()

# - LDA "each document is made up of a distribution of topics and that each topic is in turn made up of a distribution of words.
# The hidden or 'latent' layer - the topic layer" - what is it 'about'
# https://towardsdatascience.com/building-a-topic-modeling-pipeline-with-spacy-and-gensim-c5dc03ffc619
# https://alvinntnu.github.io/NTNU_ENC2045_LECTURES/nlp/topic-modeling-naive.html
# https://www.kaggle.com/thebrownviking20/topic-modelling-with-spacy-and-scikit-learn


print('5. Multi category Text Classification using spacy3 and gensim')

print('6. Text data visualiaton with Wordcloud')

#papers=['first docuemnt is great','second document is ok','second']
# import documents to papers
# wordcloud(papers)
