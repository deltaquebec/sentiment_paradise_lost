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
# get most used nouns, verbs, and adjectives in each chapter
#########################################################################
def freq_n(colm):

    # save cleaned chapters 
    chapters = []
    for i in range(12):
        chapters.append(pl_df_clean.loc[i,colm])

    # initialize lists for frequencies
    nn_freq = []
    vb_freq = []
    jj_freq = []

    # initialize text files
    pos_freq = ["freq_noun.txt","freq_verb.txt","freq_adj.txt"]

    for i in pos_freq:
        with open(i, "w+") as f:
            print("POS frequencies",file=f)

    # get pos frequencies   
    for i in chapters:
        tokens = RegexpTokenizer(r'\w+').tokenize(i)
        words = [k for k in tokens]
        tagged = pos_tag(words)

        # nouns
        nouns = [k for k, pos in tagged if (pos == 'NN')]
        Noun = FreqDist(nouns).most_common(10)
        nn_freq.append(Noun)
        with open("freq_noun.txt", "a+") as f:
            print(Noun,file=f)
            print('\n',file=f)

        # verbs    
        verbs = [k for k, pos in tagged if (pos == 'VB')]
        Verb = FreqDist(verbs).most_common(10)
        vb_freq.append(Verb)
        with open("freq_verb.txt", "a+") as f:
            print(Verb,file=f)
            print('\n',file=f)

        # adjectives       
        adjs = [k for k, pos in tagged if (pos == 'JJ')]
        Adj = FreqDist(adjs).most_common(10)
        jj_freq.append(Adj)
        with open("freq_adj.txt", "a+") as f:
            print(Adj,file=f)
            print('\n',file=f)

#########################################################################
# wordcloud definition
#########################################################################
def cloud(text):

    # extract total continuous text
    tot=' '.join(text)

    # limit word count
    wordcount = 150

    # setup wordcloud; generate wordcloud
    wordcloud = WordCloud(scale=3, background_color ='black', max_words=wordcount).generate(tot)

    # show wordclouds together
    f = plt.figure()
    
    # total reviews wordcloud
    plt.imshow(wordcloud,interpolation='bilinear')

    # annotate plot
    plt.title('Wordcloud of Poem')
    plt.axis('off')
    
    # plot
    plt.savefig("vis_data_cloud.png", dpi=300)
    plt.show(block=True)

#########################################################################
# n-gram definition
#########################################################################
def n_gram(text):
    # extract total continuous text
    tot=' '.join(text)
    
    # delineate words in text
    stringtot = tot.split(" ")
    
    # n to loop through
    gram = [1,2,3]

    # save each n-gram for plotting
    save = []

    # loop through each of tot, pos, and neg dataframes for each type of 1, 2, and 3 grams
    for i in gram:
        # look for top 15 used items
        n_gram = (pd.Series(nltk.ngrams(stringtot, i)).value_counts())[:15]
        # save as dataframe
        n_gram_df=pd.DataFrame(n_gram)
        n_gram_df = n_gram_df.reset_index()
        # aquire index, word, count
        n_gram_df = n_gram_df.rename(columns={"index": "word", 0: "count"})
        # append data to save
        save.append(n_gram_df)

    #set seaborn plotting aesthetics as default
    sns.set(rc={'figure.figsize':(11.7,8.27)})

    # define plotting region (3 rows, 3 columns)
    fig, axes = plt.subplots(3)

    # adjust space between each subplot
    plt.subplots_adjust(hspace = 0.7)

    # create barplot for each data in save
    sns.barplot(data=save[0], x='count', y='word', ax=axes[0]).set(title="1-gram for total")
    sns.barplot(data=save[1], x='count', y='word', ax=axes[1]).set(title="2-gram for total")
    sns.barplot(data=save[2], x='count', y='word', ax=axes[2]).set(title="3-gram for total")

    # plot
    plt.savefig("vis_data_n_gram.png", dpi=300)
    plt.show()

#########################################################################
# average word length definition
#########################################################################
def leng(text):
    # average word length together
    f = plt.figure()

    # get words for calculation
    word = text.str.split().apply(lambda x : [len(i) for i in x])

    # create histplot
    sns.histplot(word.map(lambda x: np.mean(x)),stat='density',kde=True,color='blue')

    # annotate plot
    plt.title("Average Word Length in Poem")
    plt.xlabel("Average Word Length")
    plt.ylabel("Probability Density")

    # plot
    plt.savefig("vis_data_leng.png", dpi=300)
    plt.show()

#########################################################################
# lexical density definition
#########################################################################
def lex_dens():

    # set up for plot space
    patch1 = mpatches.Patch(color='r',label='Normalized: 2824 tokens')
    patch2 = mpatches.Patch(color='b', label='Full token count')
    all_handles = (patch1, patch2)

    # begin plots
    fig, ax = plt.subplots()

    # adjust opacity
    ax.set_alpha(0.7)

    # create barplots
    ax.barh(pl_df_clean['chapter'], pl_df_clean['lex_dens_norm'],color='r',alpha=.5)
    ax.barh(pl_df_clean['chapter'], pl_df_clean['lex_dens'],color='b',alpha=.7)

    # annotate plots
    ax.set_title("Lexical Density by Book")
    ax.set_xlabel("Score")
    ax.set_ylabel("Book")
    ax.set_yticklabels(pl_df_clean.chapter, rotation=0)
    ax.legend(handles=all_handles,loc='lower left')
    ax.tick_params(axis='x', which='major')
    ax.invert_yaxis()

    # plot
    plt.savefig("vis_data_dens.png", dpi=300)
    plt.show()

freq_n('clean_text')
cloud(pl_df_clean['clean_text'])
n_gram(pl_df_clean['clean_text'])
leng(pl_df_clean['clean_text'])
lex_dens()

