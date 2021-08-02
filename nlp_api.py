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

#endpoint TEXT SUMMARY
#summarise using sentences with high frequency words (normalised)

# load the spacy language model into memory (the model must be installed by pip)
print('load the spacy language model into memory (the model must be installed by pip)')
nlp = spacy.load("en_core_web_lg")
print('en_core_web_lg installed')

def text_summary(text, limit):
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

#open the txt file for analysis
text=file.open()
print(text_summary('Riding the momentum and to capture the market’s imagination, she went on to showcase her creations at Malaysia Fashion Week in 2014, MICAM Shanghai, Mercedes-Benz STYLO Fashion Grand Prix, the closing winners’ party of the inaugural Kuala Lumpur City Grand Prix, the global launch of Malaysia Fashion Week in Paris, Malaysia Fashion WeeK in Kuala Lumpur and Taipei in Style in 2015 to much success. She also awarded the protégé award from Merceds-Benz STYLO Fashion award.  In May 2017, she has also showed the creations at Mercedes-Benz Stylo AsiaFashionFestival and be awarded the most promising designer of the year 2017.  In April 2018, She showed her 2018AW collection at Penang Fashion Week2018 and work with Nickeledeon’s Sponge Bob Gold. In November, She showed her 2019 SS collection at Malaysia Fashion Week 2018 and was awarded the Malaysia Fashion Week Best Accessories Designer of the Year 2018', 4))




# supervised : document classication

# unsupervised : synonym creator (keyword analysis)
