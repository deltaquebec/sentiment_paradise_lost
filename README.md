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

# text
pl_df_clean = pd.DataFrame(data=[k for k in chapters], columns=['text'])
pl_df_clean = pl_df_clean.replace('\n',' ', regex=True)
```
