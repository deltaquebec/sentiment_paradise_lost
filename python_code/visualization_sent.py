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
sent_df=pd.read_csv(r'sent_df.csv')

#########################################################################
# report average scores
#########################################################################
def avgs():
    # txt file for report
    with open("results_pl_sent.txt", "w+") as h:
        print('Statistical Results',file=h)
        print('\n',file=h)

        # VADER results
        print('VADER',file=h)
        print(sent_df[['neg','neu','pos','compound']].describe().apply(lambda x: round(x, 2)),file=h)
        print('\n',file=h)
        
        # TextBLOB results
        print('TextBLOB',file=h)
        print(sent_df[['polarity','subjectivity']].describe().apply(lambda x: round(x, 2)),file=h)
        print('\n',file=h)
        
        # NRC results
        print('NRC',file=h)
        print(sent_df[['fear','anger','anticipation','trust','surprise',
                       'positive','negative','sadness','disgust','joy']].describe().apply(lambda x: round(x, 2)),file=h)
        print('\n',file=h)
        
        
#########################################################################
# visualize sentament via VADER
#########################################################################
def vasdent():
    # define plot
    ax = plt.gca()
    # plot relevant data for each neg, neu, and pos
    sent_df.plot(kind='line',x='chapter', y='neg', color='red', alpha=.35, linewidth=5, ax=ax)
    sent_df.plot(kind='line',x='chapter', y='neu', color='blue', alpha=.35, linewidth=5, ax=ax)
    sent_df.plot(kind='line',x='chapter', y='pos', color='green', alpha=.35, linewidth=5, ax=ax)
    # set center line
    plt.axhline(y=0, xmin=0, xmax=1, alpha=.5, color='black', linestyle='--', linewidth=5)
    # set legend
    plt.legend(loc='best')
    # set title
    plt.title('Sentiment by Book (VADER)')
    # axis limits
    plt.xlim(-1,12)
    plt.ylim(-1,1)
    # axis labels
    plt.xlabel('Book')
    # x-axis annotation
    plt.xticks(np.arange(12), sent_df.chapter[0:12])
    # y-axis label
    plt.ylabel('Average Sentiment')
    # show graph
    plt.savefig("vis_sent_vader.png", dpi=300)
    plt.show()

#########################################################################
# sentiment via TextBLOB
#########################################################################
def blobpol():
    # define plot
    ax2 = plt.gca()
    # plot relevant data for polarity and subjectivity
    sent_df.plot(kind='line',x='chapter', y='polarity', color='blue', alpha=.35,linewidth=5, ax=ax2)
    sent_df.plot(kind='line',x='chapter', y='subjectivity', color='orange', alpha=.35, linewidth=5, ax=ax2)
    # set legend
    plt.legend(loc='best')
    # set title
    plt.title('Polarity/Subjectivity by Book (TextBlob)')
    # axis limits
    plt.xlim(-1,12)
    # x-axis label and annotation
    plt.xlabel('Book')
    plt.xticks(np.arange(12), sent_df.chapter[0:12])
    # y-axis label
    plt.ylabel('Average Sentiment')
    # show graph
    plt.savefig("vis_sent_blob.png", dpi=300)
    plt.show()

#########################################################################
# polarity sentiment via NRC
#########################################################################
def chapsent():
    # define plot
    ax = plt.gca()
    # plot relevant data for positive and negative sentiment
    sent_df.plot(kind='line',x='chapter', y='positive', color='green', alpha=.35, linewidth=5, ax=ax)
    sent_df.plot(kind='line',x='chapter', y='negative', color='red', alpha=.35, linewidth=5, ax=ax)
    # set legend
    plt.legend(loc='best')
    # set title
    plt.title('Sentiment by Book (NRC)')
    # axis limits
    plt.xlim(-1,12)
    # x-axis label and annotation
    plt.xlabel('Book')
    plt.xticks(np.arange(12), sent_df.chapter[0:12])
    # y-axis label and annotation
    plt.yticks()
    plt.ylabel('Sentiment Score')
    # show graph
    plt.savefig("vis_sent_nrc.png", dpi=300)
    plt.show()

#########################################################################
# negativity sentiment via NRC
#########################################################################
def chapneg():
    # define plot
    ax = plt.gca()
    # plot relevant data for negative sentiment
    sent_df.plot(kind='line',x='chapter', y='anger', color='red', alpha=.45, linewidth=5, ax=ax)
    sent_df.plot(kind='line',x='chapter', y='disgust', color='purple', alpha=.45, linewidth=5, ax=ax)
    sent_df.plot(kind='line',x='chapter', y='fear', color='maroon', alpha=.45, linewidth=5, ax=ax)
    sent_df.plot(kind='line',x='chapter', y='sadness', color='black', alpha=.45, linewidth=5, ax=ax)
    # set legend
    plt.legend(loc='best')
    # set title
    plt.title('Sentiment (Negative) by Book (NRC)')
    # axis limits
    plt.xlim(-1,12)
    # x-axis label and annotation
    plt.xlabel('Book')
    plt.xticks(np.arange(12), sent_df.chapter[0:12])
    # y-axis label and annotation
    plt.yticks()
    plt.ylabel('Sentiment Score')
    # show graph
    plt.savefig("vis_sent_nrc_neg.png", dpi=300)
    plt.show()

#########################################################################
# positivity sentiment via NRC
#########################################################################
def chappos():
    # define plot
    ax = plt.gca()
    # plot relevant data for positive sentiment
    sent_df.plot(kind='line',x='chapter', y='anticipation', color='blue', alpha=.45, linewidth=5, ax=ax)
    sent_df.plot(kind='line',x='chapter', y='joy', color='orange', alpha=.45, linewidth=5, ax=ax)
    sent_df.plot(kind='line',x='chapter', y='surprise', color='lightblue', alpha=.45, linewidth=5, ax=ax)
    sent_df.plot(kind='line',x='chapter', y='trust', color='green', alpha=.45, linewidth=5, ax=ax)
    # set legend
    plt.legend(loc='best')
    # set title
    plt.title('Sentiment (Positive) by Book (NRC)')
    # axis limits
    plt.xlim(-1,12)
    # x-axis label and annotation
    plt.xlabel('Book')
    plt.xticks(np.arange(12), sent_df.chapter[0:12])
    # y-axis label and annotation
    plt.yticks()
    plt.ylabel('Sentiment Score')
    # show graph
    plt.savefig("vis_sent_nrc_pos.png", dpi=300)
    plt.show()

#########################################################################
# diametric sadness/joy via NRC
######################################################################### 
def sadjoy():
    # define plot
    ax = plt.gca()
    # plot relevant data for positive sentiment
    sent_df.plot(kind='line',x='chapter', y='sadness', color='blue', alpha=.45, linewidth=5, ax=ax)
    sent_df.plot(kind='line',x='chapter', y='joy', color='yellow', alpha=.45, linewidth=5, ax=ax)
    # set legend
    plt.legend(loc='best')
    # set title
    plt.title('Joy/Sadness by Book (NRC)')
    # axis limits
    plt.xlim(-1,12)
    # x-axis label and annotation
    plt.xlabel('Book')
    plt.xticks(np.arange(12), sent_df.chapter[0:12])
    # y-axis label and annotation
    plt.yticks()
    plt.ylabel('Sentiment Score')
    # show graph
    plt.savefig("vis_sent_nrc_sadjoy.png", dpi=300)
    plt.show()

#########################################################################
# diametric surprise/anticipation via NRC
#########################################################################    
def surant():
    # define plot
    ax = plt.gca()
    # plot relevant data for surprise/anticipation sentiment
    sent_df.plot(kind='line',x='chapter', y='anticipation', color='orange', alpha=.45, linewidth=5, ax=ax)
    sent_df.plot(kind='line',x='chapter', y='surprise', color='teal', alpha=.45, linewidth=5, ax=ax)
    # set legend
    plt.legend(loc='best')
    # set title
    plt.title('Surprise/Anticipation by Book (NRC)')
    # axis limits
    plt.xlim(-1,12)
    # x-axis label and annotation
    plt.xlabel('Book')
    plt.xticks(np.arange(12), sent_df.chapter[0:12])
    # y-axis label and annotation
    plt.yticks()
    plt.ylabel('Sentiment Score')
    # show graph
    plt.savefig("vis_sent_nrc_surant.png", dpi=300)
    plt.show()

#########################################################################
# diametric trust/disgust via NRC
#########################################################################     
def trudis():
    # define plot
    ax = plt.gca()
    # plot relevant data for trust/disgust sentiment
    sent_df.plot(kind='line',x='chapter', y='trust', color='green', alpha=.45, linewidth=5, ax=ax)
    sent_df.plot(kind='line',x='chapter', y='disgust', color='purple', alpha=.45, linewidth=5, ax=ax)
    # set legend
    plt.legend(loc='best')
    # set title
    plt.title('Trust/Disgust by Book (NRC)')
    # axis limits
    plt.xlim(-1,12)
    # x-axis label and annotation
    plt.xlabel('Book')
    plt.xticks(np.arange(12), sent_df.chapter[0:12])
    # y-axis label and annotation
    plt.yticks()
    plt.ylabel('Sentiment Score')
    # show graph
    plt.savefig("vis_sent_nrc_trudis.png", dpi=300)
    plt.show()
    
#########################################################################
# diametric anger/fear via NRC
######################################################################### 
def feang():
    # define plot
    ax = plt.gca()
    # plot relevant data for trust/disgust sentiment
    sent_df.plot(kind='line',x='chapter', y='anger', color='red', alpha=.45, linewidth=5, ax=ax)
    sent_df.plot(kind='line',x='chapter', y='fear', color='forestgreen', alpha=.45, linewidth=5, ax=ax)
    # set legend
    plt.legend(loc='best')
    # set title
    plt.title('Anger/Fear by Book (NRC)')
    # axis limits
    plt.xlim(-1,12)
    # x-axis label and annotation
    plt.xlabel('Book')
    plt.xticks(np.arange(12), sent_df.chapter[0:12])
    # y-axis label and annotation
    plt.yticks()
    plt.ylabel('Sentiment Score')
    # show graph
    plt.savefig("vis_sent_nrc_feang.png", dpi=300)
    plt.show()

avgs()
vasdent()
blobpol()
chapsent()
chapneg()
chappos()
sadjoy()
surant()
trudis()
feang()

