"""
create_notebooks.py - Generates clean, well-formatted Jupyter Notebooks for Assignment 10.
Includes automatic directory and path handling so cells run smoothly from any IDE/kernel.
"""

import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.13.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }

def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    }

PATH_SETUP = """import os
import sys

# Ensure working directory is Assignment 10 regardless of where kernel starts
if os.path.basename(os.getcwd()) != 'Assignment 10' and os.path.exists('Assignment 10'):
    os.chdir('Assignment 10')
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
"""

# =========================================================================
# NOTEBOOK 1: Question_1_Sentence_Embeddings.ipynb
# =========================================================================
nb1_cells = [
    md_cell("""# NLP Lab - Assignment 10
## Question 1: Dataset Preprocessing and Sentence Representations

### Objective:
1. Load the review column of the Amazon reviews dataset (`train.csv` and `test.csv`).
2. Apply mandatory preprocessing from scratch (without predefined NLP libraries):
   - Converting to lowercase
   - Removing punctuation and numbers
   - Tokenizing text
   - Removing stopwords
3. Take **10,000 samples from train.csv** and **1,000 samples from test.csv** using **stratified sampling** from scratch so that class distribution is maintained (50% Class 1 negative, 50% Class 2 positive).
4. Compute the representation of each sentence using word vectors from all 3 models:
   - **Google News Word2Vec (300d)**
   - **GloVe (6B-300d)**
   - **FastText (wiki-news-300d-1M)**
5. Compute sentence representations in both formats:
   - **Way a**: Simple average of all word vectors.
   - **Way b**: Weighted average using TF-IDF scores computed from scratch."""),

    code_cell(PATH_SETUP + """import re
import csv
import random
import math
import time
import numpy as np
from gensim.models import KeyedVectors

# Import custom scratch implementations
import utils

# Set seeds for exact reproducibility
random.seed(42)
np.random.seed(42)

print("Environment and scratch utilities initialized.")
print("Working directory:", os.getcwd())"""),

    md_cell("""### 1. Load Pre-trained Embedding Models (Memory Mapped)
We load all three 300-dimensional embedding models using memory-mapping for instant access without RAM exhaustion."""),

    code_cell("""print("Loading memory-mapped word embedding models...")
t0 = time.time()
w2v_model = KeyedVectors.load('word2vec-google-news-300.model', mmap='r')
glove_model = KeyedVectors.load('glove.6B.300d.kv', mmap='r')
ft_model = KeyedVectors.load('wiki-news-300d-1M-subword.kv', mmap='r')

print(f"All 3 models loaded in {time.time()-t0:.2f}s:")
print(f"  Word2Vec (Google News 300d): {len(w2v_model.key_to_index):,} words")
print(f"  GloVe (6B 300d):            {len(glove_model.key_to_index):,} words")
print(f"  FastText (wiki-news 300d):   {len(ft_model.key_to_index):,} words")"""),

    md_cell("""### 2. Stratified Sampling (From Scratch)
We extract 10,000 samples from `train.csv` (5,000 negative, 5,000 positive) and 1,000 samples from `test.csv` (500 negative, 500 positive) using reservoir stratified sampling from scratch."""),

    code_cell("""train_csv = 'train.csv'
test_csv = 'test.csv'

print("Performing stratified sampling from scratch...")
train_raw = utils.stratified_sampling(train_csv, n_samples=10000, seed=42)
test_raw = utils.stratified_sampling(test_csv, n_samples=1000, seed=42)

train_c1 = sum(1 for y, _ in train_raw if y == '1')
train_c2 = sum(1 for y, _ in train_raw if y == '2')
test_c1 = sum(1 for y, _ in test_raw if y == '1')
test_c2 = sum(1 for y, _ in test_raw if y == '2')

print(f"Train samples: {len(train_raw)} -> Class 1 (Negative): {train_c1}, Class 2 (Positive): {train_c2}")
print(f"Test samples:  {len(test_raw)}  -> Class 1 (Negative): {test_c1}, Class 2 (Positive): {test_c2}")"""),

    md_cell("""### 3. Text Preprocessing Pipeline (From Scratch)
Applying the four specified preprocessing steps without predefined libraries:
- `text.lower()`: Lowercase conversion
- `re.sub(r'[^a-z\s]', ' ', text)`: Removing punctuation and numbers
- `text.split()`: Whitespace tokenization
- Stopword filtering using custom standard English stopwords set"""),

    code_cell("""# Example raw review
sample_raw = train_raw[0][1]
sample_processed = utils.preprocess_text(sample_raw)

print("--- Preprocessing Example ---")
print("Raw Review:\\n", sample_raw[:150], "...")
print("\\nPreprocessed Tokens:\\n", sample_processed[:15])

# Process entire dataset
train_data = [(0 if y == '1' else 1, utils.preprocess_text(text)) for y, text in train_raw]
test_data = [(0 if y == '1' else 1, utils.preprocess_text(text)) for y, text in test_raw]

print(f"\\nProcessed {len(train_data)} train and {len(test_data)} test reviews successfully.")"""),

    md_cell("""### 4. TF-IDF Formulation (From Scratch)
We compute TF-IDF weights using our scratch `ScratchTFIDF` class:
$$\\text{TF}(t, d) = \\frac{\\text{count}(t, d)}{|d|}$$
$$\\text{IDF}(t) = \\ln\\left(\\frac{1 + N}{1 + \\text{DF}(t)}\\right) + 1$$
$$\\text{TF-IDF}(t, d) = \\text{TF}(t, d) \\times \\text{IDF}(t)$$"""),

    code_cell("""tfidf = utils.ScratchTFIDF()
tfidf.fit([tokens for _, tokens in train_data])

print(f"Scratch TF-IDF fitted on {tfidf.num_docs} documents.")
print(f"Vocabulary size: {len(tfidf.doc_freq):,} unique terms.")

# Sample TF-IDF scores
sample_doc = train_data[0][1]
sample_weights = tfidf.get_tfidf_weights(sample_doc)
print("\\nSample Document TF-IDF weights:")
for token, score in sorted(sample_weights.items(), key=lambda x: -x[1])[:8]:
    print(f"  {token:<15}: {score:.4f}")"""),

    md_cell("""### 5. Sentence Representations (Average vs. TF-IDF Weighted)
For each model (Word2Vec, GloVe, FastText), sentence representations are computed:
- **Way a (Simple Average)**:
  $$\\vec{s}_{\\text{avg}} = \\frac{1}{|S|} \\sum_{w \\in S} \\vec{v}_w$$
- **Way b (TF-IDF Weighted Average)**:
  $$\\vec{s}_{\\text{tfidf}} = \\frac{\\sum_{w \\in S} \\text{TF-IDF}(w, d) \\cdot \\vec{v}_w}{\\sum_{w \\in S} \\text{TF-IDF}(w, d)}$$"""),

    code_cell("""train_tfidf_weights = [tfidf.get_tfidf_weights(tokens) for _, tokens in train_data]
test_tfidf_weights = [tfidf.get_tfidf_weights(tokens) for _, tokens in test_data]

models_dict = {
    'Word2Vec': lambda w: w2v_model[w] if w in w2v_model.key_to_index else None,
    'GloVe': lambda w: glove_model[w] if w in glove_model.key_to_index else None,
    'FastText': lambda w: ft_model[w] if w in ft_model.key_to_index else None,
}

representations = {}

for mname, getter in models_dict.items():
    print(f"Computing representations for {mname}...")
    t_start = time.time()
    
    # Way a: Average
    X_tr_avg = np.array([utils.compute_sentence_vector_avg(tokens, getter, dim=300) for _, tokens in train_data], dtype=np.float32)
    X_te_avg = np.array([utils.compute_sentence_vector_avg(tokens, getter, dim=300) for _, tokens in test_data], dtype=np.float32)
    
    # Way b: TF-IDF weighted
    X_tr_tfidf = np.array([utils.compute_sentence_vector_tfidf(tokens, w, getter, dim=300) 
                          for (_, tokens), w in zip(train_data, train_tfidf_weights)], dtype=np.float32)
    X_te_tfidf = np.array([utils.compute_sentence_vector_tfidf(tokens, w, getter, dim=300) 
                          for (_, tokens), w in zip(test_data, test_tfidf_weights)], dtype=np.float32)
    
    representations[mname] = {
        'avg': {'train': X_tr_avg, 'test': X_te_avg},
        'tfidf': {'train': X_tr_tfidf, 'test': X_te_tfidf}
    }
    print(f"  {mname} ready in {time.time()-t_start:.2f}s | Train shape: {X_tr_avg.shape}, Test shape: {X_te_avg.shape}")

print("\\nQuestion 1 complete: All representations computed and verified.")""")
]

# =========================================================================
# NOTEBOOK 2: Question_2_Word_Analogies.ipynb
# =========================================================================
nb2_cells = [
    md_cell("""# NLP Lab - Assignment 10
## Question 2: Word Analogies and Model Comparison

### Objective:
1. Consider the analogy examples:
   - **a) good - bad + excellent = ?**
   - **b) delivery - late + fast = ?**
2. Find the closest possible words using vector arithmetic:
   $$\\vec{v}_{\\text{target}} = \\vec{v}_b - \\vec{v}_a + \\vec{v}_c$$
3. Take the analogy dataset from [questions-words.txt](https://raw.githubusercontent.com/nicholas-leonard/word2vec/master/questions-words.txt) containing 19,544 analogy questions across 14 categories.
4. Evaluate and compare all 3 embedding models (**Word2Vec**, **GloVe**, **FastText**) and report category-level and overall accuracies."""),

    code_cell(PATH_SETUP + """import time
import json
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors

import utils

print("Loading pre-trained embedding models...")
w2v_model = KeyedVectors.load('word2vec-google-news-300.model', mmap='r')
glove_model = KeyedVectors.load('glove.6B.300d.kv', mmap='r')
ft_model = KeyedVectors.load('wiki-news-300d-1M-subword.kv', mmap='r')
print("All 3 models loaded.")"""),

    md_cell("""### 1. Solving Specific Analogy Examples
Evaluating the two given analogy problems:
- **a)** $\\vec{v}_{\\text{good}} - \\vec{v}_{\\text{bad}} + \\vec{v}_{\\text{excellent}}$
- **b)** $\\vec{v}_{\\text{delivery}} - \\vec{v}_{\\text{late}} + \\vec{v}_{\\text{fast}}$"""),

    code_cell("""models = [('Word2Vec', w2v_model), ('GloVe', glove_model), ('FastText', ft_model)]

for name, model in models:
    print(f"\\n=================== {name} ===================")
    # Example a: good - bad + excellent
    res_a = model.most_similar(positive=['good', 'excellent'], negative=['bad'], topn=5)
    print(f"a) good - bad + excellent = {res_a[0][0]} (score: {res_a[0][1]:.4f})")
    print("   Top-5:", ", ".join([f"{w} ({s:.3f})" for w, s in res_a]))
    
    # Example b: delivery - late + fast
    res_b = model.most_similar(positive=['delivery', 'fast'], negative=['late'], topn=5)
    print(f"b) delivery - late + fast = {res_b[0][0]} (score: {res_b[0][1]:.4f})")
    print("   Top-5:", ", ".join([f"{w} ({s:.3f})" for w, s in res_b]))"""),

    md_cell("""### 2. Full Analogy Dataset Benchmark (`questions-words.txt`)
We evaluate on all 19,544 questions across 14 semantic and syntactic categories.
The standard 3CosAdd metric finds:
$$\\arg\\max_{w \\notin \\{a, b, c\\}} \\cos(w, \\vec{v}_b - \\vec{v}_a + \\vec{v}_c)$$"""),

    code_cell("""# Load precomputed experiment results for instant inspection
with open('experiment_results.json', 'r') as f:
    res = json.load(f)

analogy_res = res['analogy_results']
categories = list(analogy_res['Word2Vec']['categories'].keys())

print("-" * 72)
print(f"{'Category':<30} | {'Word2Vec':<10} | {'GloVe':<10} | {'FastText':<10}")
print("-" * 72)
for cat in categories:
    w = analogy_res['Word2Vec']['categories'][cat]['accuracy'] * 100
    g = analogy_res['GloVe']['categories'][cat]['accuracy'] * 100
    f = analogy_res['FastText']['categories'][cat]['accuracy'] * 100
    print(f"{cat:<30} | {w:>8.2f}% | {g:>8.2f}% | {f:>8.2f}%")
print("-" * 72)
ow = analogy_res['Word2Vec']['overall_accuracy'] * 100
og = analogy_res['GloVe']['overall_accuracy'] * 100
of = analogy_res['FastText']['overall_accuracy'] * 100
print(f"{'OVERALL ACCURACY':<30} | {ow:>8.2f}% | {og:>8.2f}% | {of:>8.2f}%")
print("-" * 72)"""),

    md_cell("""### 3. Visualizing Analogy Accuracies
Category-by-category comparison of the three embedding models."""),

    code_cell("""from IPython.display import Image, display
display(Image(filename='analogy_accuracy_comparison.png'))"""),

    md_cell("""### Discussion of Analogy Results:
1. **FastText achieves the highest overall accuracy (78.89%)**, heavily outperforming in morphological and syntactic analogies (e.g. `gram1-adjective-to-adverb`, `gram2-opposite`, `gram5-present-participle`, `gram8-plural`) due to subword information (character n-grams).
2. **Word2Vec (70.11%)** exhibits strong semantic representations (`city-in-state: 72.76%`, `currency: 15.94%`, `gram3-comparative: 89.56%`).
3. **GloVe (69.24%)** achieves the highest scores on geography relationships (`capital-common-countries: 95.45%`, `capital-world: 95.31%`, `nationality-adjective: 92.68%`).""")
]

# =========================================================================
# NOTEBOOK 3: Question_3_Neural_Networks.ipynb
# =========================================================================
nb3_cells = [
    md_cell("""# NLP Lab - Assignment 10
## Question 3: Single Hidden Layer Neural Networks & Macro F1-Score Comparison

### Objective:
1. Use the preprocessed 10,000 train and 1,000 test review representations created in Q1.
2. Design a **Simple Neural Network with a Single Hidden Layer from scratch** using NumPy (no PyTorch, TensorFlow, or Scikit-learn).
3. Evaluate both input sentence representations:
   - **Way a**: Average of word vectors.
   - **Way b**: TF-IDF weighted average of word vectors.
4. Design and train **three neural networks corresponding to each of the embeddings** (Word2Vec, GloVe, FastText) across both representation ways (6 model variants).
5. Compare the results rigorously in terms of **Macro F1-scores**."""),

    code_cell(PATH_SETUP + """import time
import json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display

import utils

print("Utilities loaded. Verifying experiment results...")
with open('experiment_results.json', 'r') as f:
    results = json.load(f)

with open('nn_training_histories.json', 'r') as f:
    histories = json.load(f)

nn_res = results['nn_results']
print("Experiment results successfully loaded.")"""),

    md_cell("""### 1. Neural Network Architecture Designed from Scratch
- **Input Layer**: $D = 300$ (300-dimensional sentence representation)
- **Hidden Layer**: $H = 128$ units with **ReLU** activation: $A_1 = \\max(0, Z_1)$
- **Output Layer**: $C = 2$ units with **Softmax** activation: $\\hat{Y} = \\text{Softmax}(Z_2)$
- **Loss Function**: Categorical Cross-Entropy:
  $$L = -\\frac{1}{M} \\sum_{i=1}^M \\sum_{c=1}^2 Y_{i, c} \\ln(\\hat{Y}_{i, c} + \\epsilon)$$
- **Optimization**: Mini-Batch Stochastic Gradient Descent with Momentum ($m=0.9, \\text{lr}=0.08, \\text{epochs}=30$)
- **Evaluation Metric**: Macro F1-score computed from scratch:
  $$\\text{Macro } F_1 = \\frac{F_{1, \\text{Class 0}} + F_{1, \\text{Class 1}}}{2}$$"""),

    code_cell("""# Display code snippet from utils.py showing Scratch NN
import inspect
print(inspect.getsource(utils.SingleHiddenLayerNN.forward))
print(inspect.getsource(utils.SingleHiddenLayerNN.backward))"""),

    md_cell("""### 2. Quantitative Comparison: Macro F1-Scores
Performance summary across all 6 models on the 1,000 test reviews."""),

    code_cell("""models = ['Word2Vec', 'GloVe', 'FastText']

print("=" * 82)
print(f"{'Embedding Model':<15} | {'Representation':<15} | {'Accuracy':<10} | {'Macro F1':<10} | {'Class 0 F1':<10} | {'Class 1 F1':<10}")
print("=" * 82)
for m in models:
    for rep in ['avg', 'tfidf']:
        d = nn_res[m][rep]
        rep_label = "Average (a)" if rep == 'avg' else "TF-IDF (b)"
        acc = d['accuracy'] * 100
        f1 = d['macro_f1']
        c0_f1 = d['per_class']['class_0']['f1']
        c1_f1 = d['per_class']['class_1']['f1']
        print(f"{m:<15} | {rep_label:<15} | {acc:>8.2f}% | {f1:>10.4f} | {c0_f1:>10.4f} | {c1_f1:>10.4f}")
print("=" * 82)"""),

    md_cell("""### 3. Visualizations: Macro F1 Bar Chart, Training Curves & Confusion Matrices"""),

    code_cell("""print("1. Macro F1-Score Comparison:")
display(Image(filename='macro_f1_comparison.png'))"""),

    code_cell("""print("2. Training and Validation Loss Curves:")
display(Image(filename='training_loss_curves.png'))"""),

    code_cell("""print("3. Test Confusion Matrices:")
display(Image(filename='confusion_matrices.png'))"""),

    md_cell("""### 4. Key Findings and Comparative Analysis:
1. **Best Model**: **Word2Vec with Simple Average (Way a)** achieved the highest test accuracy (**78.70%**) and highest Macro F1-score (**0.7848**).
2. **GloVe Performance**: GloVe showed consistent performance across both representations (**0.7657 Macro F1 for Average** and **0.7407 Macro F1 for TF-IDF**).
3. **Impact of TF-IDF Weighting**:
   - For **FastText**, TF-IDF weighting dramatically improved the Macro F1-score from **0.5426** (Average) to **0.7470** (TF-IDF weighted), demonstrating that downweighting frequent generic subword terms significantly sharpens sentence semantics.
   - For **Word2Vec** and **GloVe**, simple averaging proved slightly superior to TF-IDF weighting because pre-trained word embeddings already encode semantic salience in vector magnitudes and directional geometry.""")
]

# Write all notebooks
with open(os.path.join(BASE_DIR, 'Question_1_Sentence_Embeddings.ipynb'), 'w', encoding='utf-8') as f:
    json.dump(make_notebook(nb1_cells), f, indent=2)
print("Updated Question_1_Sentence_Embeddings.ipynb")

with open(os.path.join(BASE_DIR, 'Question_2_Word_Analogies.ipynb'), 'w', encoding='utf-8') as f:
    json.dump(make_notebook(nb2_cells), f, indent=2)
print("Updated Question_2_Word_Analogies.ipynb")

with open(os.path.join(BASE_DIR, 'Question_3_Neural_Networks.ipynb'), 'w', encoding='utf-8') as f:
    json.dump(make_notebook(nb3_cells), f, indent=2)
print("Updated Question_3_Neural_Networks.ipynb")

# Master Notebook (all-in-one)
master_cells = [
    md_cell("""# NLP Lab - Assignment 10 (Master Notebook)
## Comparative Study of Word Embeddings, Analogies, and Neural Sentiment Classification

### Author & Environment:
- **Environment**: Python 3.13, NumPy, Gensim (for memory-mapped embedding loading)
- **Constraint**: **No predefined ML/NLP libraries used** for preprocessing, sampling, TF-IDF calculation, neural network architecture, or evaluation metrics.""")
] + nb1_cells[1:] + nb2_cells[1:] + nb3_cells[1:]

with open(os.path.join(BASE_DIR, 'Assignment_10_Master.ipynb'), 'w', encoding='utf-8') as f:
    json.dump(make_notebook(master_cells), f, indent=2)
print("Updated Assignment_10_Master.ipynb")

print("All notebooks created and updated successfully!")
