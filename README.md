# Data Visualization and Sentiment Analysis of "*Paradise Lost*" by John Milton
<p align="center"> 
<img src="/assets/dore_satan.jpg" alt="Satan by Dore">
</p>

The goal of this project is twofold: 1) **practice with data exploration and visualization 2) **classify sentiment** across various NLP sentiment analysis tools. The latter is achieved through [VADER](https://github.com/cjhutto/vaderSentiment), [TextBLOB](https://textblob.readthedocs.io/en/dev/), and [NRC](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm). This project follows from the work of NBrisbon on [The Silmarillion](https://github.com/NBrisbon/Silmarillion-NLP), and so frequent comparison are made.

The project is arranged as follows:

1. **Data and Lexical Information**
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

# 1. Data and Lexical Information

The tasks of data formatting and lexical analysis are contained in _dataframe_data.py_ and _dataframe_sent.py_. Visualization of the lexical information is completed in _visualization_data.py_. The plots of distributions are saved as .png files.

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

Milton's English is itself an exploration of the poet's unique mastery of language. Haynes (2000), among others, notes the poet's use of multilingual assets such as syntax and vocabulary, with deliberate attention to the narrator's and the characters' chosen rhetorical styles. His language is not Middle English, nor is it artificially archaicised like Spenser, nor is it exactly a King James biblical style. T.S. Elliot remarks that "Milton writes English like a dead language". See [here](https://jonreeve.com/2016/07/paradise-lost-macroetymology/) for an etymological study of Milton's language.

This has the interesting problem of how to clean the data for analysis. For this project, we clean the data according to English stopwords complemented with an additional series of stopwords with older English archaisms in _clean_stop.txt_. While not present in Milton's work, contractions are mapped to their periphrasis for good measure in _clean_map.txt_. The cleaning algorithm follows closely from [this work](https://github.com/072arushi/Movie_review_analysis). The cleaned text is saved to the dataframe.

```
sw = nltk.corpus.stopwords.words('english')
with open('clean_stop.txt','r') as f:
    newStopWords = f.readlines()
sw.extend(newStopWords)
pers = ['thee','thou','thy','thee','hast','u\'','thus',
            'shall','though','art','seest','knowest','shalt']
sw.extend(pers)

with open('clean_map.txt','r') as f:
    mapping = f.readlines()

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
pl_df_clean=pd.read_csv(r'pl_df_clean.csv')

def freq_n(colm):

    chapters = []
    for i in range(12):
        chapters.append(pl_df_clean.loc[i,colm])

    nn_freq = []
    vb_freq = []
    jj_freq = []

    pos_freq = ["freq_noun.txt","freq_verb.txt","freq_adj.txt"]

    for i in pos_freq:
        with open(i, "w+") as f:
            print("POS frequencies",file=f)

    for i in chapters:
        tokens = RegexpTokenizer(r'\w+').tokenize(i)
        words = [k for k in tokens]
        tagged = pos_tag(words)

        nouns = [k for k, pos in tagged if (pos == 'NN')]
        Noun = FreqDist(nouns).most_common(10)
        nn_freq.append(Noun)
        with open("freq_noun.txt", "a+") as f:
            print(Noun,file=f)
            print('\n',file=f)

        verbs = [k for k, pos in tagged if (pos == 'VB')]
        Verb = FreqDist(verbs).most_common(10)
        vb_freq.append(Verb)
        with open("freq_verb.txt", "a+") as f:
            print(Verb,file=f)
            print('\n',file=f)
            
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

[('man', 31), ('life', 23), ('death', 21), ('day', 17), ('god', 17), ('eye', 17), ('son', 16), ('till', 15), ('earth', 13), ('world', 13)]
[('god', 26), ('law', 21), ('son', 17), ('man', 15), ('nation', 15), ('death', 15), ('seed', 14), ('world', 13), ('life', 12), ('day', 12)]

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

## n-gram (mono-,bi-,tri-gram)

[n-grams](https://web.stanford.edu/~jurafsky/slp3/3.pdf) track the frquency in which (word) tokens appear. 1-grams (monograms) refer to the frequency in which single word tokens appear; 2-grams (bigrams) refer to the frequency in which two word tokens appear together; 3-grams (trigrams) refer to the frequency in which three word tokens appear together. Roughly, such frequencies will follow a [Zipf-like distribution](https://web.archive.org/web/20021010193014/http://linkage.rockefeller.edu/wli/zipf/).

We loop through 1-, 2-, and 3-gram analyses for each book of the poem, and save the top fifteen of each n-grams saved as dataframes.

```
def n_gram(text):
    
    tot=' '.join(text)
    
    stringtot = tot.split(" ")
    
    gram = [1,2,3]
    
    save = []

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

    sns.set()

    fig, axes = plt.subplots(3)

    plt.subplots_adjust(hspace = 0.7)

    sns.barplot(data=save[0], x='count', y='word', ax=axes[0]).set(title="1-gram for total")
    sns.barplot(data=save[1], x='count', y='word', ax=axes[1]).set(title="2-gram for total")
    sns.barplot(data=save[2], x='count', y='word', ax=axes[2]).set(title="3-gram for total")

    plt.savefig("vis_data_n_gram.png", dpi=300)
    plt.show()
```

n-grams are useful in that they tell us exactly word distributions (once appropriately filtered). Word pairings are exceptionally useful in many contexts, not limited to sentiment. Reconstruction of broken or incomplete texts, for example, and auto-correct are applications of n-grams. 

<p align="center"> 
<img src="/assets/vis_data_n_gram.png" alt="n_gram">
</p>

## Average word length represented as probability density

The average word length is represented by a _probability density_, the values of which may be greater than 1; the distribution itself, however, will integrate to 1. The values of the y-axis, then, are useful for relative comparisons between categories. Converting to a probability (in which the bar heights sum to 1) in the code is simply a matter of changing the argument stat='density' to stat='probability', which is essentially equivalent to finding the area under the curve for a specific interval. See [this article](https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0) for more details.

```
def leng(text):
    f = plt.figure()

    word = text.str.split().apply(lambda x : [len(i) for i in x])

    sns.histplot(word.map(lambda x: np.mean(x)),stat='density',kde=True,color='blue')

    plt.title("Average Word Length in Poem")
    plt.xlabel("Average Word Length")
    plt.ylabel("Probability Density")

    plt.savefig("vis_data_leng.png", dpi=300)
    plt.show()

```

Here, Milton is fairly consistent in his word-length for lexical words. While attempted to fit to a gaussian, the distribution follows with an average of 5.57.

<p align="center"> 
<img src="/assets/vis_data_leng.png" alt="data_leng">
</p>

## Lexical density

We now plot the lexical densities for each book of the poem. We plot both together.

```
def lex_dens():
    patch1 = mpatches.Patch(color='r',label='Normalized: 2824 tokens')
    patch2 = mpatches.Patch(color='b', label='Full token count')
    all_handles = (patch1, patch2)

    fig, ax = plt.subplots()

    ax.set_alpha(0.7)

    ax.barh(pl_df_clean['chapter'], pl_df_clean['lex_dens_norm'],color='r',alpha=.5)
    ax.barh(pl_df_clean['chapter'], pl_df_clean['lex_dens'],color='b',alpha=.7)

    ax.set_title("Lexical Density by Book")
    ax.set_xlabel("Score")
    ax.set_ylabel("Book")
    ax.set_yticklabels(pl_df_clean.chapter, rotation=0)
    ax.legend(handles=all_handles,loc='lower left')
    ax.tick_params(axis='x', which='major')
    ax.invert_yaxis()

    plt.savefig("vis_data_dens.png", dpi=300)
    plt.show()
```

Indeed, Milton is again fairly consistent, and has a relatively high inventory of lexical items. The normalized data show that Milton's choice of lexical content is consistent across books of the poem. 

<p align="center"> 
<img src="/assets/vis_data_dens.png" alt="data_dens">
</p>

# 2. Sentiment Analysis

The sentiment analysis plots are calculated in _visualization.py_ from the sentiment dataframe. The average aggregate values are printed to the text file _results_pl_sent.txt_. Much of the code for plotting sentiment is redundant, and so are not reported below.

## VADER

The results of the VADER analysis are given below. The Neutral sentiment (neu) scores higher abouve the other sentiments consistently across all books of the poem. Here we face an interesting interpretation of results. Indeed, this may follow from Milton's style, which is certainly a formal register, or it may be a limitation of the analytical application, given that the language of _Paradise Lost_ is not the intended target for VADER (this is a recurring theme, in that the deliberately archaic style of the work is quite dissimilar to the styles the sentiment analysis tools are trained on).

The trending of positive (pos) and negative (neg) scores mirror each other, While the compound score oscillates between extreme values of -1 and +1. Intersection points between positive and negative scores (Books I to III, Book VI, Books IX to XI) represent regions in the poem worth mentioning. Books I to III narrate Satan's new situation after having been expelled from heaven, his resolve to claim the exile and rallying of fallen angels to their new station, and God and the Son of God observing and conversing about Satan's actions. Book VI narrates the triumph of heaven over the rebellious angels, and Books IX to XI essentially follow the Eden story with an optimistic conclusion.

| | | **neg** | | **neu** | | **pos** | | **compound** |
|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|
| **mean**   | | 0.16 | | 0.58 | | 0.26 | | 0.67 | 
| **std**   | | 0.06 | | 0.04 | | 0.05 | | 0.78 | 
| **min**  | | 0.07 | | 0.52 | | 0.20 | | -1.00 | 
| **max**  | | 0.26 | | 0.65 | | 0.33 | | 1.00 | 

<p align="center"> 
<img src="/assets/vis_sent_vader.png" alt="sent_vader">
</p>

## TextBLOB

We see from the results below that the Subjectivity score is not only significnaly higher than the polarity score, but is also much more consistent. Interestingly, TextBLOB extrema points tend to correlate with intersection points described above in the VADER analysis, in which minima of polarity scores in TextBLOB correlate with increasing negative scores and decreasing positivy scores in VADER, and vise versa. The consistency and high scores for the Subjectivity metric indicate a more personal narrative rather than an objective account. 

| | | **polarity** | | **subjectivity** | 
|:--------|:--------|:--------|:--------|:--------|
| **mean**   | | 0.14 | | 0.53 |
| **std**   | | 0.06 | | 0.01 | 
| **min**  | | 0.05 | | 0.51 |
| **max**  | | 0.23 | | 0.55 | 

<p align="center"> 
<img src="/assets/vis_sent_blob.png" alt="sent_blob">
</p>

Comparisons and applications of the two analyses have been featured in various investigations, including two direct comarisons ([here](https://www.analyticsvidhya.com/blog/2021/10/sentiment-analysis-with-textblob-and-vader/) and [here](https://towardsdatascience.com/sentiment-analysis-vader-or-textblob-ff25514ac540)), as well as a sentiment analysis of [The Silmarillion](https://github.com/NBrisbon/Silmarillion-NLP). 

## NRC

Plutchik describes the attributes in diametric pairs, and so we plot them accordingly in addition to overall postive and negative groups (that is, we consider the positive group (anticipation, joy, surprise, and trust) relative the negative group (anger, disgust, fear, and sadness)). The NRC scores indicate that the poem tends towards a higher overall positive score, where the stringest emotions are: trust; fear; anticipation; joy. The lowest scores are: surprise; disgust; anger; sadness.

| | | **fear** | | **anger** | | **anticipation** | | **trust** | | **surprise** | | **positive** | | **negative** | | **sadness** | | **disgust** | | **joy** |
|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|
| **mean**   | | 0.11 | | 0.07 | | 0.10 | | 0.12 | | 0.04 | | 0.20 | | 0.15 | | 0.07 | | 0.05 | | 0.09 |
| **std**   | | 0.03 | | 0.02 | | 0.02 | | 0.03 | | 0.01 | | 0.04 | | 0.03 | | 0.02 | | 0.01 | | 0.02 |
| **min**  | | 0.06 | | 0.04 | | 0.07 | | 0.08 | | 0.03 | | 0.13 | | 0.09 | | 0.04 | | 0.02 | | 0.06 |
| **max**  | | 0.15 | | 0.10 | | 0.12 | | 0.17 | | 0.05 | | 0.27 | | 0.19 | | 0.11 | | 0.08 | | 0.13 |

Overall sentiment matches with the sentiment reported in the VADER analysis, and again matches the extrema points in the TextBLOB analysis. The graphs of diametric emotions are not given here, but are in the assets folder of this project.

<p align="center"> 
<img src="/assets/vis_sent_nrc.png" alt="sent_nrc">
</p>

## Conclusion

Across each sentiment analysis, we see a general trend towards positivity scores. Indeed, this is congruent with the overall optimistic conclusion of the work; even though mankind is banished from Eden, they do so having been promised of redemption. Emotional extrema in the poem correlate. Polarity extrema in TextBLOB relate as crossover points in VADER and NRC for positivity and negativity metrics, and do so such that minima of polarity scores in TextBLOB correlate with increasing negative scores and decreasing positivy scores in VADER and NRC, and vise versa. These emotional points in the narrative occur at the regions of Books I to III, Book VI, and Books IX to XI.
