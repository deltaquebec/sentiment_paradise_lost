# Data Visualization and Sentiment Analysis of "*Paradise Lost*" by John Milton
<p align="center"> 
<img src="/assets/dore_satan.jpg" alt="Satan by Dore">
</p>

The goal of this project is twofold: 1) **practice with data exploration and visualization 2) **classify sentiment** across various NLP sentiment analysis tools. The latter is achieved through [VADER](https://github.com/cjhutto/vaderSentiment), [TextBlob](https://textblob.readthedocs.io/en/dev/), and [NRC](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm). This project follows from the work of NBrisbon on [The Silmarillion](https://github.com/NBrisbon/Silmarillion-NLP), and so frequent comparison are made.

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

The tasks of data visualization are contained in a two python files: _dataframe_data.py_ and _visualization_data.py_. The plots of distributions are saved as .png files.

## Data preparation and data cleaning

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

For good measure, we also build a dataframe with a non-cleaned dataset. Here, we may include other details such as uppercase count, stopwords count, punctuation count, sentence count.

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
