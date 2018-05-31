import pickle
import pandas as pd
import numpy as np
import math

import time
from datetime import time
from datetime import datetime
import re
import string
from collections import Counter
from operator import itemgetter

from textblob import TextBlob
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation


df_agg = pickle.load(open("full_df_from_mongo_and_missing.pkl", "rb"))
print('total df size: ' +str(len(df_agg)))

#TEXT CLEANING

#remove retweets
df_noRT = df_agg[df_agg['text'].astype(str).str.startswith('RT') == False]

print('removed retweets, new length: ' + str(len(df_noRT)))
print(datetime.now().time())

#clean docs
print('begin text cleaning')

documents = np.asarray(df_noRT.text)

cleaned_docs = []
hashtags = []

for doc in documents:
    hashtag = re.compile('#[a-zA-Z0-9]*')
    match = hashtag.findall(doc)
    hashtags.append(match)
    doc = re.sub('@\S*', '', doc)
    doc = re.sub('&amp', 'and', doc)
    doc = re.sub('&amp ', 'and', doc)
    doc = re.sub('https:\S*', '', doc)
    doc = re.sub('\\n', ' ', doc)
    doc = re.sub('[0-9]', '', doc)
    exclude = set('!"$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'+'…')
    doc = ''.join(ch for ch in doc if ch not in exclude)
    cleaned_docs.append(doc)

pickle.dump( cleaned_docs, open( "run_out/cleaned_docs.pkl", "wb" ) )

## Create new dataframepickle.dump( df_cleaned_docs, open( "df_cleaned_docs_final.pkl", "wb" ) )with cleaned docs and hashtags
df_cleaned_docs = pd.DataFrame(columns = ['id', 'coords','hashtags', 'full_text', 'cleaned_text'])
df_cleaned_docs.id = df_noRT.id
df_cleaned_docs.coords = df_noRT.coords
df_cleaned_docs.hashtags = hashtags
df_cleaned_docs.full_text = df_noRT.text
df_cleaned_docs.cleaned_text = cleaned_docs



pickle.dump( df_cleaned_docs, open( "run_out/df_cleaned_docs_final.pkl", "wb" ) )

print('completed and saved clean text')
print(datetime.now().time())


#SENTIMENT

print('begin sentiment analysis')

with open('run_out/doc_sentiment.csv', 'a') as file: 
    polarity = []
    subjectivity = []
    for doc in cleaned_docs:
        sentiment = TextBlob(doc).sentiment
        doc_polarity = sentiment[0]
        doc_subjectivity = sentiment[1]
        polarity.append(doc_polarity)
        subjectivity.append(doc_subjectivity)
        file.write(str(doc_polarity) + ','+ str(doc_subjectivity) +'\n')


print('end sentiment analysis')
print(datetime.now().time())

#NER
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from itertools import groupby

print('begin NER analysis')

st = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz','stanford-ner.jar',encoding='utf-8')

entities = []
i = 0

for text in cleaned_docs:
    tokenized_text = word_tokenize(text)
    classified_text = st.tag(tokenized_text)
    doc_entities = []
    for tag, chunk in groupby(classified_text, lambda x:x[1]):
        if tag != "O":
            entity = " ".join(w for w, t in chunk)
            doc_entities.append(entity)
    entities.append(doc_entities)
    i+=1
    if i %1000 == 0:
        print(i)
        pickle.dump(entities, open( "run_out/ner_entities_list.pkl", "wb" ) )

pickle.dump(entities, open( "run_out/ner_entities_list.pkl", "wb" ) )

print('end NER analysis')
print(datetime.now().time())

print('finished. Created the following objects: 1) cleaned docs &df, 2) sentiment csv 3) entities list pkl')









