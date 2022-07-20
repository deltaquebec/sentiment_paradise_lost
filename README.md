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
- n-gram (mono-,bi-,tri-gram)
- Average word length represented as probability density
- Lexical density (including normalization)
2. **Sentiment Analysis**
- VADER
- TextBlob
- NRC

It should be noted that this project was conducted using the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit). To optimize completion time, each task was invited to perform on the GPU: NVIDIA GeFORCE RTX 3070. Preliminary tests were done on the CPU: AMD Ryzen 7 3700X 8-Core Processor 3.59 GHz. Your completion times may be different according to the processing unit you use.
