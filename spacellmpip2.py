export OPENAI_API_KEY="sk-..."
export OPENAI_API_ORG="org-..."

import spacy

nlp = spacy.blank("en")
llm = nlp.add_pipe("llm_textcat")
llm.add_label("INSULT")
llm.add_label("COMPLIMENT")
doc = nlp("You look gorgeous! im in love with it")
print(doc.cats)
# {"COMPLIMENT": 1.0, "INSULT": 0.0}