"""
utils.py - Custom implementations from scratch for Assignment 10
No predefined libraries used for preprocessing, sampling, TF-IDF, neural networks, or metrics.
"""

import math
import random
import re
import numpy as np

# Standard English stopwords defined from scratch
STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren',
    "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but',
    'by', 'can', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't",
    'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't",
    'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here',
    "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm",
    "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more',
    'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or',
    'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she',
    "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's",
    'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they',
    "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under',
    'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were',
    "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who',
    "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll",
    "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'
}


def preprocess_text(text):
    """
    Apply mandatory preprocessing steps:
    1. Converting to lowercase
    2. Removing punctuation and numbers
    3. Tokenizing text
    4. Removing stopwords
    """
    if not isinstance(text, str):
        return []
    # 1. Lowercase
    text = text.lower()
    # 2. Remove punctuation and numbers (keep only a-z and whitespace)
    text = re.sub(r'[^a-z\s]', ' ', text)
    # 3. Tokenize by whitespace
    tokens = text.split()
    # 4. Remove stopwords and empty/single-char tokens
    cleaned = [w for w in tokens if w not in STOPWORDS and len(w) > 1]
    return cleaned


def stratified_sampling(csv_path, n_samples, seed=42):
    """
    Stratified sampling from scratch maintaining balanced class distribution.
    For binary labels '1' (negative) and '2' (positive), selects n_samples // 2
    from each class uniformly at random without external libraries.
    """
    import csv
    rng = random.Random(seed)
    n_per_class = n_samples // 2
    
    pools = {'1': [], '2': []}
    counts = {'1': 0, '2': 0}
    
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            lbl = row[0].strip()
            if lbl not in pools:
                continue
            text = row[2].strip()
            counts[lbl] += 1
            
            # Reservoir sampling per class
            if len(pools[lbl]) < n_per_class:
                pools[lbl].append((lbl, text))
            else:
                j = rng.randint(0, counts[lbl] - 1)
                if j < n_per_class:
                    pools[lbl][j] = (lbl, text)
            
            # Stop early once sufficient statistical coverage is achieved
            if counts['1'] >= n_per_class * 10 and counts['2'] >= n_per_class * 10:
                break
                
    combined = pools['1'] + pools['2']
    rng.shuffle(combined)
    return combined


class ScratchTFIDF:
    """
    TF-IDF Vectorizer implemented completely from scratch.
    TF(t, d) = count(t, d) / total_words_in_d
    IDF(t) = ln((1 + N) / (1 + DF(t))) + 1  (smooth IDF)
    """
    def __init__(self):
        self.doc_freq = {}
        self.idf = {}
        self.num_docs = 0
        
    def fit(self, tokenized_corpus):
        self.num_docs = len(tokenized_corpus)
        self.doc_freq = {}
        for doc in tokenized_corpus:
            seen = set(doc)
            for token in seen:
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1
                
        # Compute smooth IDF: ln((1 + N) / (1 + DF)) + 1
        for token, df in self.doc_freq.items():
            self.idf[token] = math.log((1 + self.num_docs) / (1 + df)) + 1.0
            
    def get_tfidf_weights(self, doc_tokens):
        """
        Returns a dict of token -> tfidf_score for a document
        """
        if not doc_tokens:
            return {}
        counts = {}
        for t in doc_tokens:
            counts[t] = counts.get(t, 0) + 1
        total_len = len(doc_tokens)
        
        tfidf = {}
        # default idf for OOV words in idf dictionary
        default_idf = math.log((1 + self.num_docs) / 1.0) + 1.0
        for t, cnt in counts.items():
            tf = cnt / float(total_len)
            idf = self.idf.get(t, default_idf)
            tfidf[t] = tf * idf
        return tfidf


def compute_sentence_vector_avg(tokens, embedding_getter, dim=300):
    """
    Way a: The sentence representation is the average of all the word vectors in the sentence.
    """
    vectors = []
    for token in tokens:
        v = embedding_getter(token)
        if v is not None:
            vectors.append(v)
            
    if not vectors:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vectors, axis=0).astype(np.float32)


def compute_sentence_vector_tfidf(tokens, tfidf_dict, embedding_getter, dim=300):
    """
    Way b: The sentence representation is the weighted average of all the word vectors
    in the sentence where the weights are the TF-IDF scores of the words.
    """
    vectors = []
    weights = []
    for token in tokens:
        v = embedding_getter(token)
        if v is not None:
            w = tfidf_dict.get(token, 1.0)
            vectors.append(v)
            weights.append(w)
            
    if not vectors:
        return np.zeros(dim, dtype=np.float32)
        
    weights = np.array(weights, dtype=np.float32)
    sum_w = np.sum(weights)
    if sum_w <= 1e-8:
        return np.mean(vectors, axis=0).astype(np.float32)
        
    weights = weights / sum_w
    weighted_sum = np.sum([w * v for w, v in zip(weights, vectors)], axis=0)
    return weighted_sum.astype(np.float32)


def cosine_similarity(u, v):
    """
    Cosine similarity computed from scratch: (u . v) / (||u|| * ||v||)
    """
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u < 1e-9 or norm_v < 1e-9:
        return 0.0
    return float(np.dot(u, v) / (norm_u * norm_v))


class SingleHiddenLayerNN:
    """
    Simple Neural Network with a Single Hidden Layer designed from scratch using NumPy.
    Architecture:
      Input (dim 300) -> Hidden Layer (128 units, ReLU activation) -> Output Layer (2 units, Softmax)
    Trained with Mini-Batch Gradient Descent and Cross-Entropy Loss.
    """
    def __init__(self, input_dim=300, hidden_dim=128, output_dim=2, seed=42):
        rng = np.random.RandomState(seed)
        # He / Xavier initialization
        self.W1 = rng.randn(input_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim), dtype=np.float32)
        
        self.W2 = rng.randn(hidden_dim, output_dim).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim), dtype=np.float32)
        
        # Momentum buffers
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)
        
    def _relu(self, z):
        return np.maximum(0.0, z)
        
    def _softmax(self, z):
        # Numerically stable softmax
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self._relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.probs = self._softmax(self.Z2)
        return self.probs
        
    def compute_loss(self, Y_true, probs):
        m = Y_true.shape[0]
        eps = 1e-15
        loss = -np.sum(Y_true * np.log(probs + eps)) / m
        return float(loss)
        
    def backward(self, X, Y_true, lr=0.01, momentum=0.9, l2_reg=1e-4):
        m = X.shape[0]
        # Gradient on output layer
        dZ2 = (self.probs - Y_true) / float(m)
        dW2 = np.dot(self.A1.T, dZ2) + l2_reg * self.W2
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        # Gradient on hidden layer
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (self.Z1 > 0).astype(np.float32)
        dW1 = np.dot(X.T, dZ1) + l2_reg * self.W1
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        # SGD with Momentum update
        self.v_W1 = momentum * self.v_W1 + lr * dW1
        self.v_b1 = momentum * self.v_b1 + lr * db1
        self.v_W2 = momentum * self.v_W2 + lr * dW2
        self.v_b2 = momentum * self.v_b2 + lr * db2
        
        self.W1 -= self.v_W1
        self.b1 -= self.v_b1
        self.W2 -= self.v_W2
        self.b2 -= self.v_b2
        
    def fit(self, X, y, X_val=None, y_val=None, epochs=25, batch_size=64, lr=0.05, verbose=True):
        """
        Fit model using Mini-Batch SGD.
        y is a 1D array of labels 0 or 1.
        """
        m = X.shape[0]
        # One-hot encode targets
        Y_onehot = np.zeros((m, 2), dtype=np.float32)
        Y_onehot[np.arange(m), y] = 1.0
        
        if y_val is not None and X_val is not None:
            m_val = X_val.shape[0]
            Y_val_onehot = np.zeros((m_val, 2), dtype=np.float32)
            Y_val_onehot[np.arange(m_val), y_val] = 1.0
            
        history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
        
        for epoch in range(epochs):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            Y_shuffled = Y_onehot[indices]
            
            for start in range(0, m, batch_size):
                end = min(start + batch_size, m)
                batch_X = X_shuffled[start:end]
                batch_Y = Y_shuffled[start:end]
                
                self.forward(batch_X)
                self.backward(batch_X, batch_Y, lr=lr)
                
            train_probs = self.forward(X)
            train_loss = self.compute_loss(Y_onehot, train_probs)
            history['train_loss'].append(train_loss)
            
            if X_val is not None and y_val is not None:
                val_probs = self.forward(X_val)
                val_loss = self.compute_loss(Y_val_onehot, val_probs)
                val_preds = np.argmax(val_probs, axis=1)
                metrics = compute_macro_f1(y_val, val_preds)
                history['val_loss'].append(val_loss)
                history['val_f1'].append(metrics['macro_f1'])
                
                if verbose and (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"Epoch {epoch+1:2d}/{epochs:2d} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Macro F1: {metrics['macro_f1']:.4f}")
            else:
                if verbose and (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"Epoch {epoch+1:2d}/{epochs:2d} - Train Loss: {train_loss:.4f}")
                    
        return history
        
    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)


def compute_confusion_matrix(y_true, y_pred, num_classes=2):
    """
    Confusion matrix from scratch.
    cm[i, j] is number of true class i predicted as class j.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def compute_macro_f1(y_true, y_pred):
    """
    Compute Precision, Recall, and Macro F1-score from scratch.
    """
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=2)
    # cm:
    #      pred 0   pred 1
    # true 0  TN     FP
    # true 1  FN     TP
    f1_list = []
    metrics_per_class = {}
    
    for c in range(2):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        
        prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / float(prec + rec) if (prec + rec) > 0 else 0.0
        
        f1_list.append(f1)
        metrics_per_class[f'class_{c}'] = {'precision': prec, 'recall': rec, 'f1': f1}
        
    macro_f1 = float(np.mean(f1_list))
    total_acc = float(np.sum(np.diag(cm)) / np.sum(cm))
    
    return {
        'confusion_matrix': cm,
        'accuracy': total_acc,
        'macro_f1': macro_f1,
        'per_class': metrics_per_class
    }

