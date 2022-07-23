# Data Visualization and Sentiment Analysis of "*Paradise Lost*" by John Milton
<p align="center"> 
<img src="/assets/dore_satan.jpg" alt="Satan by Dore">
</p>

The goal of this project is twofold: 1) **practice with data exploration and visualization 2) **classify sentiment** across various NLP sentiment analysis tools. The latter is achieved through [VADER](https://github.com/cjhutto/vaderSentiment), [TextBLOB](https://textblob.readthedocs.io/en/dev/), and [NRC](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm). This project follows from the work of NBrisbon on [The Silmarillion](https://github.com/NBrisbon/Silmarillion-NLP), and so frequent comparison are made.

The project is arranged as follows:

1. **Visualization**
- Data preparation and data cleaning
- POS frequencies
- Wordcloud representations
- n-gram (mono-, bi-, tri-gram)
- Average word length represented as probability density
- Lexical density (including normalization)
2. **Sentiment Analysis**
- VADER
- TextBlob
- NRC

It should be noted that this project was conducted using the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit). To optimize completion time, each task was invited to perform on the GPU: NVIDIA GeFORCE RTX 3070. Preliminary tests were done on the CPU: AMD Ryzen 7 3700X 8-Core Processor 3.59 GHz. Your completion times may be different according to the processing unit you use.

# 1. Visualization

The tasks of data visualization are contained in _visualization_data.py_ and _visualization_sent.py_. The plots of distributions are saved as .png files.

## Data preparation and data cleaning

Data preparation follows from processing the text as well as preparing dataframes of numerical information as well as sentiment information. We prepare the data in _dataframe_data.py_ and the sentiment in _dataframe_sent.py_.

### Textual Data

A searchable etext of _Paradise Lost_ is available [here](https://www.paradiselost.org/8-search.html). This text was prepared with modernized spellings of Milton's archaic vocabulary to better facilitate search and analysis. Each book of the poem is saved as a text file _ch_01.txt_, _ch_02.txt_, and so on. Each txt file is accessed and saved to a dataframe with appropriate chapter titles associated with each text. Note that the dataframe is specified as "clean", as two dataframes will be made available; the raw dataframe will be the text without processing, but no further analysis will be conducted there.

```
path = r'C:\Users\delta\AppData\Local\Programs\Python\Python37\nlp\sentiment_pl\chapters'

chapters = []

def readchap(folder):
    files = glob(folder+"/*.txt") # read all text files
    for i in files :
        infile = open(i,"r",encoding="utf-8")
        data = infile.read()
        infile.close()
        chapters.append([data])
    return chapters

readchap(path)

chapter_name = ["~ BOOK I ~", "~ BOOK II ~", "~ BOOK III ~", "~ BOOK IV ~", "~ BOOK V ~", "~ BOOK VI ~",
            "~ BOOK VII ~", "~ BOOK VIII ~", "~ BOOK IX ~", "~ BOOK X ~", "~ BOOK XI ~", "~ BOOK XII ~"]

pl_df_clean = pd.DataFrame(data=[k for k in chapters], columns=['text'])
pl_df_clean = pl_df_clean.replace('\n',' ', regex=True)

chapter_name = pd.DataFrame(chapter_name, columns=['chapter'])
pl_df_clean = pd.merge(pl_df_clean, chapter_name, left_index=True, right_index=True)
```

Milton's English is itself an exploration of th epoet's unique mastery of language. Haynes (2000), among others, notes the poet's use of multilingual assets such as syntax and vocabulary, with deliberate attention to the narrator's and the character's chosen rhetorical styles. His language is not Middle English, nor is it artificially archaicised like Spenser, nor is it exactly King James biblical style. T.S. Elliot remarks that "Milton writes English like a dead language". See [here](https://jonreeve.com/2016/07/paradise-lost-macroetymology/) for an etymological study of Milton's language.

This has the interesting problem of how to clean the data for analysis. For this project, we clean the data according to English stopwords complemented with an additional series of stopwords with older English archaisms in _clean_stop.txt_. While not present in Milton's work, contractions are mapped to their periphrasis for good measure in _clean_map.txt_. The cleaning algorithm follows closely from [this work](https://github.com/072arushi/Movie_review_analysis). The cleaned text is saved to the dataframe.

```
# stopwords
sw = nltk.corpus.stopwords.words('english')
with open('clean_stop.txt','r') as f:
    newStopWords = f.readlines()
sw.extend(newStopWords)

# contractions
with open('clean_map.txt','r') as f:
    mapping = f.readlines()

# clean
punct = string.punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', punct))
def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text
def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in sw])
def word_replace(text):
    return text.replace('<br />','')
stemmer = PorterStemmer()
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])
lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)
def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)
def preprocess(text):
    text=clean_contractions(text,mapping)
    text=text.lower()
    text=word_replace(text)
    text=remove_urls(text)
    text=remove_html(text)
    text=remove_stopwords(text)
    text=remove_punctuation(text)
    text=lemmatize_words(text)
    return text

pl_df_clean['clean_text'] = pl_df_clean['text'].apply(lambda text: preprocess(text))
pl_df_clean = pl_df_clean.replace('\n',' ', regex=True)
```

Cleaned chapters are saved as a list such that they may be readily accessed.

```
chapters_clean = []
for i in range(12):
    chapters_clean.append(pl_df_clean.loc[i,'clean_text'])
```

Lexical densitity (or lexical diversity) is related as the lexical words (content words such as nouns, adjectives, adverbs, verbs) divided by the total number of words. This relationship indicates the content words per total in a text. This number can be normailzed by using the size of the smallest text (word count 2824 --- the length of the shortest chapter) instead of the true total. This allows for a scaled comparison.

```
def lexical_density(text):
    return len(set(text)) / len(text)

dens = []
for i in chapters_clean:
    tokens = RegexpTokenizer(r'\w+').tokenize(i)    
    lex_dens = lexical_density(tokens)
    dens.append(lex_dens)
lex_dens = pd.DataFrame(dens) 
lex_dens = lex_dens.rename(columns={0: 'lex_dens'})

dens_norm = []
for i in chapters_clean:
    tokens = RegexpTokenizer(r'\w+').tokenize(i)
    tokens = tokens[0:2824]     
    lex_dens_norm = lexical_density(tokens)
    dens_norm.append(lex_dens_norm)  
lex_dens_norm = pd.DataFrame(dens_norm) 
lex_dens_norm = lex_dens_norm.rename(columns={0: 'lex_dens_norm'})
```

We then compile this information into a dataframe that contains information such as: character count; word count; unique word count; average word length; lexical density; normalized lexical density. This is all saved to a .csv file such that we may analyze it at a later time.

```
pl_df_clean['char_count']=pl_df_clean['clean_text'].str.len()
pl_df_clean['word_count']=pl_df_clean['clean_text'].apply(lambda x: len(str(x).split()))
pl_df_clean['unique_count']=pl_df_clean['clean_text'].apply(lambda x: len(set(str(x).split())))
pl_df_clean['avg_word_len']=pl_df_clean['clean_text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
pl_df_clean = pd.merge(pl_df_clean, lex_dens, left_index=True, right_index=True)
pl_df_clean = pd.merge(pl_df_clean, lex_dens_norm, left_index=True, right_index=True)
pl_df_clean.to_csv(r'pl_df_clean.csv')
```

For good measure, we also build a dataframe with a non-cleaned dataset. Here, we may include other details such as uppercase count, stopwords count, punctuation count, sentence count. Note that the lexical densities are still calculated raltive to the cleaned data.

```
pl_df_raw = pd.DataFrame(data=[k for k in chapters], columns=['text'])
pl_df_raw = pl_df_raw.replace('\n',' ', regex=True)
chapter_name = pd.DataFrame(chapter_name, columns=['chapter'])
pl_df_raw = pd.merge(pl_df_raw, chapter_name, left_index=True, right_index=True)
pl_df_raw['char_count']=pl_df_raw['text'].str.len()
pl_df_raw['word_count']=pl_df_raw['text'].apply(lambda x: len(str(x).split()))
pl_df_raw['unique_count']=pl_df_raw['text'].apply(lambda x: len(set(str(x).split())))
pl_df_raw['uppercase_count']=pl_df_raw['text'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
pl_df_raw['stopwords_count']=pl_df_raw['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in sw]))
pl_df_raw['avg_word_len']=pl_df_raw['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
pl_df_raw['punct_count']=pl_df_raw['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
pl_df_raw['sent_count']=pl_df_raw['text'].apply(lambda x: len(re.findall(r"[?!.]",str(x)))+1)
pl_df_raw = pd.merge(pl_df_raw, lex_dens, left_index=True, right_index=True)
pl_df_raw = pd.merge(pl_df_raw, lex_dens_norm, left_index=True, right_index=True)
pl_df_raw.to_csv(r'pl_df_raw.csv')
```

### Sentiment Data

Now armed with the text in a dataframe, we may now prepare for sentiment analysis by accessing that dataframe and building on it. For the purposes of this project, only the cleaned text is considered. Three sentiment analysis models are considered: [VADER](https://github.com/cjhutto/vaderSentiment); [TextBLOB](https://textblob.readthedocs.io/en/dev/);  [NRC](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm).

We begin with first invoking the dataframe.

```
pl_df_clean=pd.read_csv(r'pl_df_clean.csv')
```

VADER (Valence Aware Dictionary and sEntiment Reasoner) specifically examines sentiments expressed in social media, and is sensitive to **polarity** and **intensity**. We define the VADER analyzer, and apply it to the cleaned text using a lambda expression to delineate positive, negative, and neutral sentiment, as well as compound, which is a “normalized, weighted, composite score”.

```
analyzer = SentimentIntensityAnalyzer()

pl_df_clean['vader'] = pl_df_clean['clean_text'].apply(lambda x: analyzer.polarity_scores(x))

pl_df_clean = pd.concat([pl_df_clean.drop(['vader'], axis = 1), pl_df_clean['vader'].apply(pd.Series)], axis = 1)
pl_df_clean['vader_eval'] = pl_df_clean['compound'].apply(lambda x: 'pos' if x >0 else 'neg' if x <0 else 'neu')
```

TextBLOB "is a Python library for processing textual data and helps with tasks such as part-of-speech tagging, noun phrase extraction, sentiment analysis, and more". Here, we measure a text for its **polarity** (the extent to which sentiment is positive or negative on a [-1,+1] scale) and **subjectivity** (measured on a [0,+1] scale in which 0 refers to an objective statement and 1 refers to a subjective statement; subjectivity refers to personal opinions, emotions, or judgments, while objectivity refers to factual information).

We define a list to contain sentences bassed into the model, and save lists for polarity and subjectivity.

```
blobs = []

for i in range(12):
    blobs.append(TextBlob(pl_df_clean.loc[i,'clean_text']))

blob_polr = []
blob_subj = []
```

We iterate over each blobbed text and get polarity and subjectivity scores therein. We pass those measurements into the dataframe.

```
for i in blobs:
    sent = i.sentences
    for j in sent:
        polr = j.sentiment.polarity
        subj = j.sentiment.subjectivity
    blob_polr.append(polr)
    blob_subj.append(subj)
    
polr_df = pd.DataFrame(blob_polr)  
subj_df = pd.DataFrame(blob_subj)  
blob_sent = pd.merge(polr_df, subj_df, left_index=True, right_index=True)

blob_sent = blob_sent.rename(columns={'0_x': "polarity", '0_y': 'subjectivity'})
pl_df_clean = pd.merge(pl_df_clean, blob_sent, left_index=True, right_index=True)
```

NRC measures English words and their associations with eight basic emotions and two sentiments, which combine to yield more complex human emotions. This follows from the work of psychologist and professor [Robert Plutchik](https://en.wikipedia.org/wiki/Robert_Plutchik). The ten attributes are:

- anger
- fear
- anticipation
- trust
- surprise
- sadness
- joy
- disgust
- positive
- negative

We apply the NRC model to the clean text and save the data as a dataframe.

```
pl_df_clean['emotions'] = pl_df_clean['clean_text'].apply(lambda x: NRCLex(x).affect_frequencies)

pl_df_clean = pd.concat([pl_df_clean.drop(['emotions'], axis = 1), pl_df_clean['emotions'].apply(pd.Series)], axis = 1)
```

The sentiment data and the textual data are then ready for visualization.

```
pl_df_clean.to_csv(r'sent_df.csv')
```

## POS Frequencies

Even with a modernized normalization of Milton's langauge, cataloging parts of speech such as nouns, verbs, and adjectives remains difficult. For superlative adjectives, for example, the model can confuse second person verbal inflection marker {-est) as the indicated for superlative adjectives. So while we may catalogue frequently occuring parts of speech, some error inevitably intrudes. 

We define the frequency function and iterate over chapters, saving the data to .txt files.

```
# get the data
pl_df_clean=pd.read_csv(r'pl_df_clean.csv')

# stopwords
sw = nltk.corpus.stopwords.words('english')
with open('clean_stop.txt','r') as f:
    newStopWords = f.readlines()
sw.extend(newStopWords)

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
        words = [k for k in tokens if k not in sw]
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
        with open("freq_verb.txt", "a+") as g:
            print(Verb,file=g)
            print('\n',file=g)

        # adjectives       
        adjs = [k for k, pos in tagged if (pos == 'JJ')]
        Adj = FreqDist(adjs).most_common(10)
        jj_freq.append(Adj)
        with open("freq_adj.txt", "a+") as f:
            print(Adj,file=f)
            print('\n',file=f)
```

[('hell', 16), ('god', 14), ('power', 14), ('spirit', 12), ('heaven', 12), ('fire', 12), ('force', 12), ('hath', 11), ('strength', 10), ('seat', 9)]

[('hell', 24), ('fire', 19), ('war', 18), ('heaven', 18), ('pain', 17), ('way', 17), ('power', 13), ('night', 13), ('place', 12), ('hand', 11)]

Not surprisingly, "hell" is the most represented nominal. Even glancing at these two lists, we can get a glimpse at the subject matter therein. Compare these two with the last two chapters' nominals, in which the focus is far from Satan's new situation, and instead concerns the expulsion from Eden and what is yet to come.

[('man', 31), ('life', 23), ('death', 21), ('day', 17), ('god', 17), ('eye', 17), ('son', 16), ('till', 14), ('world', 13), ('way', 11)]

[('god', 25), ('law', 21), ('son', 17), ('man', 15), ('nation', 15), ('death', 15), ('seed', 14), ('world', 13), ('life', 12), ('day', 12)]

** Wordcloud representations

Wordclouds are visualizations of (text) data in which the size of a word represents its frequency or importance in that data. Wordclouds are handy for visualization-at-a-glance, and have the enjoyable consequence of making a report more lively. 

Generating wordclouds for each of poem's books follows from joining the text into a continuous string and defining the numerical limiter for how many words will be considered after sifting through stopwords. We plot accordingly

```
def cloud(text):

    tot=' '.join(text)

    wordcount = 150

    sw = nltk.corpus.stopwords.words('english')
    with open('clean_stop.txt','r') as f:
        newStopWords = f.readlines()
    sw.extend(newStopWords)
    pers = ['thee','thou','thy']
    sw.extend(pers)
    
    wordcloud = WordCloud(scale=3, background_color ='black', max_words=wordcount, stopwords=sw).generate(tot)

    f = plt.figure()

    plt.imshow(wordcloud,interpolation='bilinear')

    plt.title('Wordcloud of Poem')
    plt.axis('off')

    plt.savefig("vis_data_cloud.png", dpi=300)
    plt.show(block=True)
```

A cursory inspection of the wordcloud can give hint as to the subject matter of the text.

<p align="center"> 
<img src="/assets/vis_data_cloud.png" alt="Wordcloud">
</p>
