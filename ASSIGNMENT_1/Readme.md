
# ASSIGNMENT 1

[![python](https://img.shields.io/badge/Python-black?style=flat-square&logo=python)](https://www.python.org/) 


This assignment implements a text tokenization (Sentence Tokenization and Word Tokenization) that converts raw text into individual tokens (words) for Natural Language Processing (NLP).

Here Tokenization is done for GUJARATI Language.
## Problem / Task

Q1.You need to complete 4 tasks.

a. Visit https://huggingface.co/datasets/ai4bharat/IndicCorpV2 website and download the data from your language. Extract all the data.

b. You need to write codes for a sentence tokenizer and word tokenizer. Tokenize each paragraph into sentences and words. Tokenize each word. Your tokenizer should tokenize punctuations, URLS, numbers (handle decimals), mail ids, dates.

c. After your data is tokenized, save them into a file or multiple files.

d. Then compute the following corpus statistics:

    i. Total number of sentences
    ii. Total number of words
    iii. Total number of characters
    iv. Average Sentence Length (Average number of words per sentence)
    v. Average word length (Average number of characters per word)
    vi. Type/Token

Q2. Repeat the same steps on a huge monolingual corpora available at https://huggingface.co/datasets/oscar-corpus/OSCAR-2301
## Folder Structure



```
ASSIGNMENT_1/
│
├── README.md                          # Project documentation
│
└── Q1/
    ├── Tokenizer.py                   # Main script for sentence and word tokenization
    ├── parquet_to_txt.py              # Utility script to convert Parquet files to CSV/TXT/Excel
    ├── tokenized_gujarati_corpus/     # Output directory containing generated Parquet files
    │   ├── gujarati_tokenized_batch_0001.parquet 
    │   ├── gujarati_tokenized_batch_0002.parquet
    │   └── ...
    └── tempCodeRunnerFile.py          # Temporary VS Code file (can be ignored)
```

**NOTE : Currently output file are of total 10000 sentences. But Code is Updated for 11 lakh sentences.Due larger output file we have uploaded sample output files**

## Approach

The pipeline converts raw Gujarati text into a tokenized, structured corpus in a few steps:

**Data Ingestion** – Sentences are read from a local text file instead of streaming from a Hugging Face dataset, making the pipeline fully offline.
User can read data by streaming.for that use:

dataset = load_dataset("ai4bharat/IndicCorpV2", "indiccorp_v2", split="guj_Gujr", streaming=True)

**Sentence Splitting** – Text is split into sentences using ., !, ? as boundaries, while skipping these characters when they appear inside URLs, emails, dates, or decimal numbers .

**Tokenization with Type Tagging** – Each sentence is tokenized using regex, with priority given to composite patterns (URL, email, date, decimal) before generic ones (number, Gujarati word, punctuation). This keeps tokens like 15-08-2026 or 3.14 intact as single tokens, and each token is tagged with its type (url, email, date, decimal, number, gujarati, punct).

**Batch Storage** – Sentences are processed in batches and saved as Parquet files (snappy compression), storing original text, tokens, token types, and category-wise breakdowns.
Statistics – Corpus-level stats (total sentences, words, characters, avg. sentence/word length, Type-Token Ratio).

**Configuration Parameters :**

To process more data:

**max_sentences** — controls how many sentences are processed. Set to None to process the entire corpus instead of a fixed limit.

**batch_size** — controls how many sentences go into each Parquet file. Lower it to get more, smaller files; raise it to get fewer, larger files.
## Libraries Used


**re** :    Performs text matching and tokenization using Regular Expressions. It is used for detecting sentences, Gujarati words, URLs, emails, dates, numbers, decimals, and punctuation. 

**os** : Handles file and directory operations such as creating output folders and generating file paths. 

**pandas** : Creates and manages DataFrames to organize the tokenized data before saving it. 

 **pyarrow** :  Provides support for the Apache Arrow format and enables efficient conversion between Pandas DataFrames and Parquet files. 
**pyarrow.parquet** : Reads and writes data in Parquet format for efficient storage and faster access. 

**collections.Counter** *(optional)* :Counts the frequency of tokens or words.  
## Installation

Install the required external libraries using pip:

```bash
pip install pandas pyarrow
```

or

```bash
python -m pip install pandas pyarrow
```

### Built-in Python Libraries (No Installation Required)

The following libraries are included with Python and do not need to be installed separately:

- `re`
- `os`
- `collections`
## Output

for  max_sentences=10000,batch_size=1000 

The tokenizer generated the following statistics for the processed Gujarati corpus:



| Metric | Value |
|--------|------:|
| Total number of sentences | **9,999** |
| Total number of words | **149,247** |
| Total number of characters | **657,508** |
| Average sentence length | **14.93 words** |
| Average word length | **4.41 characters** |
| Type/Token Ratio (TTR) | **0.1959** |
| Number of batches created | **10** |


for  max_sentences=1100000,batch_size=100000

The tokenizer generated the following statistics for the processed Gujarati corpus:



| Metric | Value |
|--------|------:|
| Total number of sentences | **1,099,721** |
| Total number of words | **16,351,576** |
| Total number of characters | **71,695,431** |
| Average sentence length | **14.87 words** |
| Average word length | **4.38 characters** |
| Type/Token Ratio (TTR) | **0.0315** |
| Number of batches created | **11** |
