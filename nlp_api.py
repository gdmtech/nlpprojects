# NLP test API
# David Williams
# Aug 1, 2021
# using Spacy - spaCy is an open-source software library for
# advanced natural language processing, written in the
# programming languages Python and Cython.

import spacy
# count frequency
from collections import Counter
# most commmon punctuation in English
from string import punctuation

# endpoint TEXT SUMMARY
# summarise using sentences with high frequency words (normalised)

# load the spacy language model into memory (the model must be installed by pip)
print('load the spacy language model into memory (the model must be installed by pip)')
nlp = spacy.load("en_core_web_lg")
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


# open the txt file for analysis
print('COGNIAI NLP DEMO')
print('Document Summary')
text = open("samples/crime.txt", "r")
print(mojo_text_summary(text.read(), 10))

print('Document Classification : rommance, drama, comedy, etc')
