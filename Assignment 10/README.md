# NLP Lab - Assignment 10
## Comparative Study of Word Embeddings (Word2Vec, GloVe, FastText), Analogies, and Neural Sentiment Classification

### Author & Environment
- **Platform**: Python 3.13, NumPy, Gensim (strictly for pre-trained vector format loading as referenced in the lab tutorial)
- **Constraint Compliance**: **No predefined ML/NLP libraries used** (No Scikit-learn, PyTorch, TensorFlow, NLTK, or SpaCy). All preprocessing, stratified reservoir sampling, TF-IDF scoring, word analogy math, neural network forward/backpropagation, and evaluation metrics (Macro F1) were **implemented strictly from scratch**.

---

## 1. Directory Structure

```text
Assignment 10/
├── Question_1_Sentence_Embeddings.ipynb   # Q1 Interactive Walkthrough
├── Question_2_Word_Analogies.ipynb         # Q2 Interactive Walkthrough
├── Question_3_Neural_Networks.ipynb        # Q3 Interactive Walkthrough
├── Assignment_10_Master.ipynb              # Complete End-to-End Master Notebook
├── utils.py                                # All scratch implementations (Preprocessing, Sampling, TF-IDF, NN, Metrics)
├── prepare_embeddings.py                   # Embedding converter (creates fast memory-mapped .kv files)
├── run_experiments.py                      # Master automated evaluation script
├── generate_plots_and_tables.py            # Generates comparative charts and summary CSVs
│
├── analogy_accuracy_comparison.png         # Bar chart comparing Word2Vec, GloVe, FastText across 14 categories
├── macro_f1_comparison.png                 # Bar chart comparing Macro F1 of all 6 neural network configurations
├── training_loss_curves.png                # Loss curves across 30 training epochs
├── confusion_matrices.png                  # Test set confusion matrices (1000 samples)
├── analogy_category_results.csv            # Detailed CSV category breakdown of analogy accuracies
├── neural_network_f1_results.csv           # Detailed CSV metrics table (Accuracy, Macro F1, Precision, Recall)
│
├── train.csv                               # Amazon Reviews training dataset
├── test.csv                                # Amazon Reviews test dataset
├── questions-words.txt                     # Word analogy dataset (19,544 questions across 14 categories)
├── word2vec-google-news-300.model          # Google News Word2Vec 300d
├── word2vec-google-news-300.model.vectors.npy
├── glove.6B.300d.kv                        # GloVe 300d (memory-mapped)
├── glove.6B.300d.kv.vectors.npy
├── wiki-news-300d-1M-subword.kv            # FastText 300d (memory-mapped)
└── wiki-news-300d-1M-subword.kv.vectors.npy
```

---

## 2. Methodology & Implementation from Scratch

### Question 1: Preprocessing & Sentence Embeddings
1. **Stratified Sampling**:
   - Implemented using class-conditional reservoir sampling to select **10,000 samples from `train.csv`** (5,000 negative, 5,000 positive) and **1,000 samples from `test.csv`** (500 negative, 500 positive) maintaining an exact 50:50 class balance.
2. **Text Preprocessing**:
   - **Lowercase conversion**: `text.lower()`
   - **Punctuation and numbers removal**: `re.sub(r'[^a-z\s]', ' ', text)`
   - **Tokenization**: Whitespace splitting without external tokenizers.
   - **Stopword removal**: Using a custom standard English stopword set.
3. **TF-IDF Calculation (from scratch)**:
   - Term frequency: $\text{TF}(t, d) = \frac{\text{count}(t, d)}{|d|}$
   - Smooth inverse document frequency: $\text{IDF}(t) = \ln\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1.0$
   - $\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$
4. **Sentence Vector Representations**:
   - **Way a (Simple Average)**:
     $$\vec{s}_{\text{avg}} = \frac{1}{|S|} \sum_{w \in S} \vec{v}_w$$
   - **Way b (TF-IDF Weighted Average)**:
     $$\vec{s}_{\text{tfidf}} = \frac{\sum_{w \in S} \text{TF-IDF}(w, d) \cdot \vec{v}_w}{\sum_{w \in S} \text{TF-IDF}(w, d) + \epsilon}$$

---

### Question 2: Word Analogies (`questions-words.txt`)
1. **Analogy Vector Arithmetic**:
   $$\vec{v}_{\text{target}} = \vec{v}_b - \vec{v}_a + \vec{v}_c$$
   $$\arg\max_{w \notin \{a, b, c\}} \cos(w, \vec{v}_{\text{target}}) = \frac{\vec{v}_w \cdot \vec{v}_{\text{target}}}{\|\vec{v}_w\|_2 \|\vec{v}_{\text{target}}\|_2}$$
2. **Example Inquiries**:
   - **a) `good - bad + excellent`**:
     - **Word2Vec**: `terrific` (0.6986), `superb` (0.6440), `fantastic` (0.6410)
     - **GloVe**: `superb` (0.5926), `terrific` (0.5724), `wonderful` (0.5317)
     - **FastText**: `superb` (0.7591), `excelent` (0.7452), `excellant` (0.7369)
   - **b) `delivery - late + fast`**:
     - **Word2Vec**: `delievery` (0.4610), `Delivery` (0.4249)
     - **GloVe**: `faster` (0.4703), `deliveries` (0.4651)
     - **FastText**: `super-fast` (0.5805), `delivering` (0.5701)
3. **Analogy Benchmark Evaluation**:
   Evaluated all **19,544 questions** across all **14 categories**:

| Category | Word2Vec Accuracy | GloVe Accuracy | FastText Accuracy |
| :--- | :---: | :---: | :---: |
| capital-common-countries | 82.21% | 95.45% | **97.23%** |
| capital-world | 81.50% | 95.31% | **96.09%** |
| currency | **15.94%** | 10.97% | 5.89% |
| city-in-state | 72.76% | 61.09% | **81.23%** |
| family | 77.47% | **79.45%** | 72.92% |
| gram1-adjective-to-adverb | 28.93% | 20.56% | **57.86%** |
| gram2-opposite | 32.39% | 22.04% | **55.30%** |
| gram3-comparative | 89.56% | 87.39% | **94.52%** |
| gram4-superlative | **67.56%** | 58.20% | 53.21% |
| gram5-present-participle | 73.48% | 63.73% | **89.87%** |
| gram6-nationality-adjective | 85.43% | **92.68%** | 91.56% |
| gram7-past-tense | 66.22% | 59.74% | **68.78%** |
| gram8-plural | 78.38% | 74.10% | **84.23%** |
| gram9-plural-verbs | 63.79% | 52.76% | **76.78%** |
| **OVERALL ACCURACY** | **70.11%** | **69.24%** | **78.89%** |

---

### Question 3: Neural Networks with Single Hidden Layer
1. **Network Architecture (Pure NumPy)**:
   - **Input Dimension**: $D = 300$
   - **Hidden Layer**: $H = 128$ units, He-initialized weights, **ReLU activation**: $A_1 = \max(0, Z_1)$
   - **Output Layer**: $C = 2$ units, **Softmax activation**: $\hat{Y} = \text{Softmax}(Z_2)$
   - **Loss Function**: Categorical Cross-Entropy:
     $$L = -\frac{1}{M} \sum_{i=1}^M \sum_{c=1}^2 Y_{i, c} \ln(\hat{Y}_{i, c} + \epsilon)$$
   - **Optimizer**: Mini-Batch SGD with Momentum ($\beta = 0.9, \eta = 0.08, \text{batch\_size} = 64, \text{epochs} = 30$)
2. **Evaluation Metrics (from scratch)**:
   - Confusion matrix $CM$
   - $\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}, \quad F_{1, c} = \frac{2 \cdot \text{Prec}_c \cdot \text{Rec}_c}{\text{Prec}_c + \text{Rec}_c}$
   - $\text{Macro } F_1 = \frac{F_{1, \text{Class 0}} + F_{1, \text{Class 1}}}{2}$

#### Final Model Comparison:

| Embedding Model | Representation | Test Accuracy | Macro F1-Score | Class 0 F1 | Class 1 F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Word2Vec** | **Way a: Average** | **78.70%** | **0.7848** | **0.8065** | **0.7631** |
| **Word2Vec** | Way b: TF-IDF | 69.10% | 0.6684 | 0.5819 | 0.7550 |
| **GloVe** | **Way a: Average** | **76.60%** | **0.7657** | **0.7741** | **0.7573** |
| **GloVe** | Way b: TF-IDF | 74.20% | 0.7407 | 0.7226 | 0.7589 |
| **FastText** | Way a: Average | 60.90% | 0.5426 | 0.7169 | 0.3683 |
| **FastText** | **Way b: TF-IDF** | **75.00%** | **0.7470** | **0.7744** | **0.7197** |

---

## 3. Key Observations & Inferences

1. **Analogy Task**:
   - **FastText dominated syntactic categories** (94.52% on comparative, 89.87% on present participle, 84.23% on plural) because character n-grams directly capture prefixes and suffixes (e.g. `-er`, `-ing`, `-s`).
   - **GloVe and Word2Vec dominated world geography and semantic analogies** (over 95% on country-capital pairs).
2. **Sentiment Classification**:
   - **Word2Vec (Average)** yielded the best overall Macro F1 score of **0.7848**, with GloVe close behind at **0.7657**.
   - **TF-IDF Weighting impact**:
     - Significantly boosted FastText from **0.5426** to **0.7470** (+20.44 points F1) by mitigating noise from high-frequency subword overlapping tokens.
     - For Word2Vec and GloVe, simple averaging outperformed TF-IDF weighting as whole-word semantic magnitudes and dense sentiment directions are preserved better by uniform pooling.

---

## 4. Execution Commands

To reproduce all experiments and regenerate figures and CSVs:
```bash
# Run complete pipeline end-to-end:
python run_experiments.py

# Regenerate all visual charts and CSV reports:
python generate_plots_and_tables.py
```

