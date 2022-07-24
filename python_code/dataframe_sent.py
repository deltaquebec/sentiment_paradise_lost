import nltk
import io
import re
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.text import Text
from nltk.corpus import brown
import requests
nltk.download('popular')
import string 
import numpy as np
from PIL import Image
from os import path
from pathlib import Path
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import pandas as pd
from nltk import tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
from textblob import TextBlob
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.stem.snowball import SnowballStemmer
from nrclex import NRCLex
import seaborn as sns
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import matplotlib.patches as mpatches

# get the data 
pl_df_clean=pd.read_csv(r'pl_df_clean.csv')

#########################################################################
# vader sentiment
#########################################################################
# define analyzer
analyzer = SentimentIntensityAnalyzer()

# obtain polarity scores
pl_df_clean['vader'] = pl_df_clean['clean_text'].apply(lambda x: analyzer.polarity_scores(x))

# split scores into distinct entries of dataframe
pl_df_clean = pd.concat([pl_df_clean.drop(['vader'], axis = 1), pl_df_clean['vader'].apply(pd.Series)], axis = 1)
pl_df_clean['vader_eval'] = pl_df_clean['compound'].apply(lambda x: 'pos' if x >0 else 'neg' if x <0 else 'neu')

#########################################################################
# blob sentiment
#########################################################################
# empty list for saving blobbed sentences
blobs = []

# blobbify text such that we may work with textblob
for i in range(12):
    blobs.append(TextBlob(pl_df_clean.loc[i,'clean_text']))

# emtpy lissts to store polarity and subjectivity scores
blob_polr = []
blob_subj = []

# iterate over each blobbed text and get polarity and subjectivity scores therein
for i in blobs:
    sent = i.sentences
    for j in sent:
        polr = j.sentiment.polarity
        subj = j.sentiment.subjectivity
    blob_polr.append(polr)
    blob_subj.append(subj)

# put scores into dataframe
polr_df = pd.DataFrame(blob_polr)  
subj_df = pd.DataFrame(blob_subj)  
blob_sent = pd.merge(polr_df, subj_df, left_index=True, right_index=True)

# rename dataframe columns and merge
blob_sent = blob_sent.rename(columns={'0_x': "polarity", '0_y': 'subjectivity'})
pl_df_clean = pd.merge(pl_df_clean, blob_sent, left_index=True, right_index=True)

#########################################################################
# nrc sentiment
#########################################################################
# apply nrc to text
pl_df_clean['emotions'] = pl_df_clean['clean_text'].apply(lambda x: NRCLex(x).affect_frequencies)

# split scores into distinct entries of dataframe
pl_df_clean = pd.concat([pl_df_clean.drop(['emotions'], axis = 1), pl_df_clean['emotions'].apply(pd.Series)], axis = 1)

# convert to and save as csv
pl_df_clean.to_csv(r'sent_df.csv')


