# KEYWORD MATCHING (P0) - NOT NEEDED FOR LAUNCH
# @app.route('/find_keyword', methods=['POST'])
# def SearchKeyword():
# CLEAN KEYWORD, GET SIMILAR TERM
# get the body from json request
#    connection = pymysql.connect(host='localhost',
#                             user='uc5rqajvtxtyy',
#                             password='xfwbkk9qx65c',
#                             db='dbw4xc5v7eaxz8',
#                             charset='utf8mb4',
#                             cursorclass=pymysql.cursors.DictCursor)
#    cursor = connection.cursor()
#
#    #get json data
#    originalSentences = request.json.get('KEYWORD', None)
#    sentence = nlp(originalSentences)

#    clean_word = []
#    enriched_sentence = []

# For each token in the sentence
#    for token in sentence:
# clean words
#        if not token.is_stop and not token.is_punct and token.tag_ != 'VBP':
# get lemma
#            clean_word.append(str(token.text.lower()))
#            clean_word.extend(x for x in lemmatizer(token.lemma_.lower(),ADJ) if x not in clean_word)
#            clean_word.extend(x for x in lemmatizer(token.lemma_.lower(),NOUN) if x not in clean_word)
#            clean_words = nlp(' '.join(clean_word))
# get synonym from domain
#            for word in clean_words:
#                synsets = word._.wordnet.wordnet_synsets_for_domain(domains)
#                lemmas_for_synset = []
#                for s in synsets:
#                    lemmas_for_synset.extend(s.lemma_names())
#                if synsets:
#                    enriched_sentence.extend(set(lemmas_for_synset))
#    cw = list(set(clean_word))
#    es = list(set([str(x) for x in nlp(' '.join(enriched_sentence).replace("_"," "))]))

# save all keywords
#    search_string = cw
#    search_string.extend(x.lower() for x in es if x not in search_string)
#    search_string = tuple(search_string)
