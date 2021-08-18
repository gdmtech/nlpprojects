# NLP test API
# David Williams
# Aug 1, 2021
# using Spacy - spaCy is an open-source software library for
# advanced natural language processing, written in the
# programming languages Python and Cython.

# explosion AI spacy 3.0 contains many new features

# Depedencies
# ml_datasets

from wordcloud import WordCloud
from ml_datasets import imdb
import spacy
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

# endpoint TEXT SUMMARY
# summarise using sentences with high frequency words (normalised)

# load the spacy language model into memory (the model must be installed by pip as its large)
# the nlp contains parsing pipelines - check with v
print('load the spacy language model into memory (the model must be installed by pip)')
nlp = spacy.load("en_core_web_lg")
print('pipelines', nlp.pipe_names)
print('en_core_web_lg installed')


# END POINT MOJO TEXT SUMMARY


def mojo_text_summary(text, limit):
    keyword = []
    # define the categories of words that we are interested in using spacy terms langauge model terms
    pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
    # create a doc (a list of tokens) from an nlp analysis of the text in lower case
    doc = nlp(text.lower())
    # create a list of tokens from the
    for token in doc:  # 2
        if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
            continue  # 3
        if(token.pos_ in pos_tag):
            keyword.append(token.text)
    print('keyword', keyword)
    # retur a list of count the number of occurances of each word
    freq_word = Counter(keyword)
    print('freq', freq_word)
    # find which words occurs the most
    max_freq = Counter(keyword).most_common(1)[0][1]
    print('max_freq', max_freq)
    # calculate the frequency from 0-1
    for w in freq_word:
        freq_word[w] = (freq_word[w]/max_freq)
    print('freq', freq_word)
    # find the sentences in the Doc that contain the top keywords
    sent_strength = {}
    for sent in doc.sents:
        for word in sent:
            if word.text in freq_word.keys():
                if sent in sent_strength.keys():
                    sent_strength[sent] += freq_word[word.text]
                else:
                    sent_strength[sent] = freq_word[word.text]
    # sort sentences in order of importance
    summary = []

    sorted_x = sorted(sent_strength.items(),
                      key=lambda kv: kv[1], reverse=True)
    # create a summary with the key sentences
    counter = 0
    for i in range(len(sorted_x)):
        # capitalise the first word of each sentence
        summary.append(str(sorted_x[i][0]).capitalize())

        counter += 1
        if(counter >= limit):
            break

    return ' '.join(summary)


def mojo_text_parse(text):

    tokens = []
    words = []
    sents = []
    stop_words = []
    filtered = []

    print("summarise the text: words, sentences, removed stops, lemmas")
    # preprocess: reduce the text to all lower cases
    doc = nlp(text.lower())
    # preprocess: tokens
    for token in doc:
        tokens.append(token.text)

    print('Tokens of all kinds', tokens)

    print('Pipleines - what do we have', tokens)

    print('Pipelines', nlp.pipe_names)
    print('Add sentencizer')

    nlp.add_pipe('sentencizer')
    print('Pipelines', nlp.pipe_names)

    # list all the sentences in the text - sentenzier has already parsed the text to doc.sents
    for sent in doc.sents:
        sents.append(sent.text)
    print(sents)

    print('remove stop words and classify text')
    for word in doc:
        # is_stop is a spacy function
        if word.is_stop == False or word.pos_ != 'PUNCT':
            filtered.append(word)
            # word,pos_ classifies the Part Of Speech (POS) of the text, word.lemma_ lists the related workds
            print(word.text, word.pos_)
            print(word.text, word.lemma_)
    print("Filtered Sentence:", filtered)

    print('entity analysis')
    # create a 3 part tuple using a for loop of attricutes of each entiy in the text
    entities = [(i, i.label_, i.label) for i in doc.ents]
    print('Entities', entities)
    # use the spacy renderer to serve a web page of the DEPENDENCIES in the text
    spacy.displacy.serve(doc, style="dep")


def text_classifier():

    # textcat single label and multi-label
    # create and add the textcat pipeline with a CNN classifier acrchitecture
    print(nlp.pipe_names)
    config = {"threshold": 0.5, "model": "spacy.TextCatCNN.v2"}

    nlp.add_pipe("textcat", config=config)
    print(nlp.pipe_names)
    print('setup the classes')

    # Adding the class labels sequentially to textcat with add_lebel method
    # textcat.add_label("POSITIVE")
    # textcat.add_label("NEGATIVE")
    print('import the data')
    # load the labelled data to train the model
    # the data must have postive or negatiev retains as output
    reviews = pd.read_csv("databases/reviews.csv")
    #df_amazon = pd.read_csv ("datasets/amazon_alexa.tsv", sep="\t")
    # we are only interested in  REview text and Recommend IND; drop columns with missing balues
    reviews = reviews[['Review Text', 'Review Index']].dropna()
    print(reviews.head(10))
    # apply the lamda function to each row ie make a tuple; apply by rows (axis=1)
    reviews['tuples'] = reviews.apply(lambda row: (
        row['Review Text'], row['Review Index']), axis=1)
    # convert to a list for training
    train = reviews['tuples'].tolist()
    print(train[:10])


def spacystopwords(limit):
    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

    # Printing the total number of stop words:
    print('Number of stop words:', len(spacy_stopwords))

    # Printing first limit stop words:
    print('First stop words:', list(spacy_stopwords)[:limit])


def text_classifier_s3():
    # https://medium.com/analytics-vidhya/building-a-text-classifier-with-spacy-3-0-dd16e9979a
    # We want to classify movie reviews as positive or negative
    # http://ai.stanford.edu/~amaas/data/sentiment/

    # we already imported IMPD
    # load movie reviews as a tuple (text, label) - the database is already configured for training
    train_data, valid_data = imdb()
    print('train_data', train_data)
    print('valid_data', valid_data)
    num_texts = 100
    # first we need to transform all the training data into annoated docs using the nlp model
    print('create the spacy docbin for training')
    train_docs = make_docs(train_data[:num_texts])
    # then we save it in a binary file to disc using the DocBin compression (similar to pickle)
    doc_bin = DocBin(docs=train_docs)
    doc_bin.to_disk("./data/train.spacy")
    # repeat for validation data
    print('create the spacy docbin for validation')
    valid_docs = make_docs(valid_data[:num_texts])
    doc_bin = DocBin(docs=valid_docs)
    doc_bin.to_disk("./data/valid.spacy")
    # the files are now ready to be used in Spacy traing


def deployed_textcat(model_file):
    # load thebest model from training
    nlp = spacy.load(model_file)
    text = ""
    print("type : ‘quit’ to exit")
    # predict the sentiment until someone writes quit
    while text != "quit":
        text = input("Please enter example input: ")
        doc = nlp(text)
        if doc.cats['positive'] > .5:
            print("the sentiment is positive")
        else:
            print("the sentiment is negative")


def make_docs(data):
    """
    this will take a list of texts and labels 
    and transform them in spacy documents

    data: list(tuple(text, label))

    returns: List(spacy.Doc.doc)
    """

    docs = []
    # nlp.pipe([texts]) is way faster than running
    # nlp(text) for each text
    # as_tuples allows us to pass in a tuple,
    # the first one is treated as text
    # the second one will get returned as it is.
    # display as progress bar
    # generate annotated data from the the nlp model
    for doc, label in tqdm(nlp.pipe(data, as_tuples=True), total=len(data)):
        # we need to set the (text)cat(egory) for each document
        #print('doc.cats >', label)
        #cats = [{"POSITIVE": bool(y), "NEGATIVE": not bool(y)} for y in labels]
        #doc.cats = {'postive':bool, 'negative':bool}
        # cats need consistent names
        if label == 'pos':
            doc.cats["positive"] = 1
            doc.cats["negative"] = 0
        else:
            doc.cats["positive"] = 0
            doc.cats["negative"] = 1
        # put them into a nice list
        docs.append(doc)
    return docs

    # we are so far only interested in the first 5000 reviews
    # this will keep the training time short.
    # In practice take as much data as you can get.
    # you can always reduce it to make the   script even faster.

def topic_model(text):
    # Parameters tuning using Grid Search

    #Preprocessing Stage 
    grid_params = {'n_components' : list(range(5,10))}


    # LDA model from scikitlearn
    lda = LatentDirichletAllocation()
    lda_model = GridSearchCV(lda,param_grid=grid_params)
    lda_model.fit(document_term_matrix)
    # Estimators for LDA model
    lda_model1 = lda_model.best_estimator_
    print("Best LDA model's params" , lda_model.best_params_)
    print("Best log likelihood Score for the LDA model",lda_model.best_score_)
    print("LDA model Perplexity on train data", lda_model1.perplexity(document_term_matrix))


    #visualise result
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
print('Examining word vectors - "castle"')
# assign  the launguage model for a word
castle = nlp("castle")
print(castle.vector.shape)
print(castle.vector)
print('0. List Spacy Stop Words')
spacystopwords(30)
print('1. Text Summary using Word Frequency Measures')
#text_file = open("samples/crime.txt", "r")
#text = text_file.read()
text = 'How about running a trip to Tokyo?  Dont be shy.  Or perhaps Kyoto or London. Nevertheless, challenges await you if you run it.  An the Financial Times will be interested.  Soon!'
print(text)
print(mojo_text_summary(text, 10))
print('2. Text Parsing : words, lemmas, entities')
#print(text)
# print(mojo_text_parse(text))
print('3. Binary Text Classification (Sentiment Analysis) using new spacy3 text_cat model + binary data import + quickstart config')
# https://www.machinelearningplus.com/nlp/custom-text-classification-spacy/
# text_classifier_s3()
deployed_textcat("./output/model-best")
print('4. Topic Modelling using spacy3 and scikitlearn')
text=''
print (text)
topic_model(text)

# - LDA "each document is made up of a distribution of topics and that each topic is in turn made up of a distribution of words.
# The hidden or 'latent' layer - the topic layer" - what is it 'about'
# https://towardsdatascience.com/building-a-topic-modeling-pipeline-with-spacy-and-gensim-c5dc03ffc619
# https://alvinntnu.github.io/NTNU_ENC2045_LECTURES/nlp/topic-modeling-naive.html
# https://www.kaggle.com/thebrownviking20/topic-modelling-with-spacy-and-scikit-learn


print('5. Multi category Text Classification using spacy3 and gensim')

print ('6. Text data visualiaton with Wordcloud')
# papers[]
# papers[0]='first docuemnt is great'
#papers[1]='second document is ok'
#import documents to papers
#wordcloud(papers)
