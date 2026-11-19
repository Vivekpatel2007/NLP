"""
run_experiments.py - Complete evaluation for Assignment 10
Executes Questions 1, 2, and 3 without predefined ML/NLP libraries.
"""

import os
import sys
import time
import json
import numpy as np
from gensim.models import KeyedVectors

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("=========================================================")
print("           NLP LAB ASSIGNMENT 10 EXPERIMENTS             ")
print("=========================================================")

# -------------------------------------------------------------------------
# STEP 1: LOAD EMBEDDINGS (Memory Mapped)
# -------------------------------------------------------------------------
print("\n[Step 1] Loading pre-trained embeddings (Memory Mapped)...")
t0 = time.time()
w2v_model = KeyedVectors.load(os.path.join(BASE_DIR, 'word2vec-google-news-300.model'), mmap='r')
glove_model = KeyedVectors.load(os.path.join(BASE_DIR, 'glove.6B.300d.kv'), mmap='r')
ft_model = KeyedVectors.load(os.path.join(BASE_DIR, 'wiki-news-300d-1M-subword.kv'), mmap='r')
print(f"Loaded all 3 models in {time.time()-t0:.2f}s:")
print(f"  - Word2Vec: {len(w2v_model.key_to_index):,} vocabulary")
print(f"  - GloVe:    {len(glove_model.key_to_index):,} vocabulary")
print(f"  - FastText: {len(ft_model.key_to_index):,} vocabulary")

def get_w2v_vec(word):
    if word in w2v_model.key_to_index:
        return w2v_model[word]
    return None

def get_glove_vec(word):
    if word in glove_model.key_to_index:
        return glove_model[word]
    return None

def get_ft_vec(word):
    if word in ft_model.key_to_index:
        return ft_model[word]
    return None

models_dict = {
    'Word2Vec': get_w2v_vec,
    'GloVe': get_glove_vec,
    'FastText': get_ft_vec
}

# -------------------------------------------------------------------------
# QUESTION 1: DATA LOADING, STRATIFIED SAMPLING & PREPROCESSING
# -------------------------------------------------------------------------
print("\n" + "="*50)
print(" QUESTION 1: PREPROCESSING & SENTENCE EMBEDDINGS")
print("="*50)

train_csv = os.path.join(BASE_DIR, 'train.csv')
test_csv = os.path.join(BASE_DIR, 'test.csv')

print("Performing stratified sampling (from scratch)...")
# 10,000 train samples (5000 class 1, 5000 class 2)
# 1,000 test samples (500 class 1, 500 class 2)
train_raw = utils.stratified_sampling(train_csv, n_samples=10000, seed=42)
test_raw = utils.stratified_sampling(test_csv, n_samples=1000, seed=42)

print(f"Sampled {len(train_raw)} train samples. Class 1: {sum(1 for y, _ in train_raw if y=='1')}, Class 2: {sum(1 for y, _ in train_raw if y=='2')}")
print(f"Sampled {len(test_raw)} test samples. Class 1: {sum(1 for y, _ in test_raw if y=='1')}, Class 2: {sum(1 for y, _ in test_raw if y=='2')}")

print("\nApplying preprocessing pipeline (lowercase, remove punct & numbers, tokenize, remove stopwords)...")
train_data = []
for y_str, text in train_raw:
    lbl = 0 if y_str == '1' else 1  # 0: negative, 1: positive
    tokens = utils.preprocess_text(text)
    train_data.append((lbl, tokens))

test_data = []
for y_str, text in test_raw:
    lbl = 0 if y_str == '1' else 1
    tokens = utils.preprocess_text(text)
    test_data.append((lbl, tokens))

print("Preprocessing complete!")
print("Sample raw text:\n ", train_raw[0][1][:120], "...")
print("Sample preprocessed tokens:\n ", train_data[0][1][:12])

# Fit TF-IDF from scratch on training tokens
print("\nFitting ScratchTFIDF model on train corpus...")
tfidf_model = utils.ScratchTFIDF()
tfidf_model.fit([tokens for _, tokens in train_data])
print(f"TF-IDF fitted on {tfidf_model.num_docs} documents, vocabulary size: {len(tfidf_model.doc_freq)}")

# Precompute TF-IDF weights for each doc
train_tfidf_weights = [tfidf_model.get_tfidf_weights(tokens) for _, tokens in train_data]
test_tfidf_weights = [tfidf_model.get_tfidf_weights(tokens) for _, tokens in test_data]

y_train = np.array([lbl for lbl, _ in train_data], dtype=np.int32)
y_test = np.array([lbl for lbl, _ in test_data], dtype=np.int32)

# Build representations for all 3 models in both ways (avg and tfidf)
representations = {}
for mname, getter in models_dict.items():
    print(f"\nBuilding sentence representations for {mname}...")
    t_start = time.time()
    # Way a: Average
    X_train_avg = np.array([utils.compute_sentence_vector_avg(tokens, getter, dim=300) for _, tokens in train_data], dtype=np.float32)
    X_test_avg = np.array([utils.compute_sentence_vector_avg(tokens, getter, dim=300) for _, tokens in test_data], dtype=np.float32)
    
    # Way b: TF-IDF weighted average
    X_train_tfidf = np.array([utils.compute_sentence_vector_tfidf(tokens, w_dict, getter, dim=300) 
                              for (_, tokens), w_dict in zip(train_data, train_tfidf_weights)], dtype=np.float32)
    X_test_tfidf = np.array([utils.compute_sentence_vector_tfidf(tokens, w_dict, getter, dim=300) 
                             for (_, tokens), w_dict in zip(test_data, test_tfidf_weights)], dtype=np.float32)
    
    representations[mname] = {
        'avg': {'train': X_train_avg, 'test': X_test_avg},
        'tfidf': {'train': X_train_tfidf, 'test': X_test_tfidf}
    }
    print(f"  {mname} representations computed in {time.time()-t_start:.2f}s (Train shape: {X_train_avg.shape}, Test shape: {X_test_avg.shape})")

# -------------------------------------------------------------------------
# QUESTION 2: WORD ANALOGIES & ACCURACY EVALUATION
# -------------------------------------------------------------------------
print("\n" + "="*50)
print(" QUESTION 2: WORD ANALOGIES EVALUATION")
print("="*50)

# Example a: good - bad + excellent = ?
# Example b: delivery - late + fast = ?
print("\nEvaluating Example Analogies:")
print("a) good - bad + excellent = ? (target vector = v(good) - v(bad) + v(excellent))")
print("b) delivery - late + fast = ? (target vector = v(delivery) - v(late) + v(fast))")

# For fast candidate search, build candidate search vocab:
# Mikolov's standard analogy search ranks among the most frequent ~30,000 words.
# We extract top 30,000 words from each model for exact cosine ranking.
TOP_CANDIDATES = 30000

for mname, model in [('Word2Vec', w2v_model), ('GloVe', glove_model), ('FastText', ft_model)]:
    print(f"\n--- Model: {mname} ---")
    
    # a) good - bad + excellent
    # Note: in vector math, "good - bad + excellent" = pos: good, excellent; neg: bad
    try:
        closest_a = model.most_similar(positive=['good', 'excellent'], negative=['bad'], topn=5)
        print(f"a) good - bad + excellent => Top prediction: '{closest_a[0][0]}' (score: {closest_a[0][1]:.4f})")
        print(f"   Top-5: {', '.join([f'{w} ({s:.3f})' for w, s in closest_a])}")
    except Exception as e:
        print(f"a) Error: {e}")
        
    # b) delivery - late + fast
    try:
        closest_b = model.most_similar(positive=['delivery', 'fast'], negative=['late'], topn=5)
        print(f"b) delivery - late + fast => Top prediction: '{closest_b[0][0]}' (score: {closest_b[0][1]:.4f})")
        print(f"   Top-5: {', '.join([f'{w} ({s:.3f})' for w, s in closest_b])}")
    except Exception as e:
        print(f"b) Error: {e}")

# Evaluating on questions-words.txt
print("\nEvaluating entire analogy dataset (questions-words.txt)...")
analogy_path = os.path.join(BASE_DIR, 'questions-words.txt')
categories = {}
current_cat = "default"
total_questions = 0

with open(analogy_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith(':'):
            current_cat = line[1:].strip()
            categories[current_cat] = []
        else:
            parts = line.split()
            if len(parts) == 4:
                categories[current_cat].append(parts)
                total_questions += 1

print(f"Loaded {total_questions} analogy questions across {len(categories)} categories.")

# Fast, vectorized analogy evaluator from scratch
# Evaluates questions where all 4 words are in model vocabulary.
# Standard 3CosAdd: argmax_{w not in {a, b, c}} cos(w, v_b - v_a + v_c)
def evaluate_analogies(model, categories_dict, candidate_limit=30000):
    # Candidate vocabulary matrix
    cand_words = list(model.key_to_index.keys())[:candidate_limit]
    cand_word_to_id = {w: i for i, w in enumerate(cand_words)}
    cand_matrix = np.array([model[w] for w in cand_words], dtype=np.float32)
    # Normalize rows
    norms = np.linalg.norm(cand_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    cand_matrix_norm = cand_matrix / norms
    
    results = {}
    total_correct = 0
    total_tested = 0
    
    for cat_name, questions in categories_dict.items():
        cat_correct = 0
        cat_tested = 0
        for w1, w2, w3, w4 in questions:
            # Check if all words in model
            # Note: lowercasing for glove/fasttext if needed
            w1_q = w1 if w1 in model.key_to_index else w1.lower()
            w2_q = w2 if w2 in model.key_to_index else w2.lower()
            w3_q = w3 if w3 in model.key_to_index else w3.lower()
            w4_target = w4.lower()
            
            if (w1_q not in model.key_to_index or 
                w2_q not in model.key_to_index or 
                w3_q not in model.key_to_index):
                continue
                
            # Target vector: v(w2) - v(w1) + v(w3)
            v_target = model[w2_q] - model[w1_q] + model[w3_q]
            v_norm = np.linalg.norm(v_target)
            if v_norm == 0:
                continue
            v_target_norm = v_target / v_norm
            
            # Cosine similarities with all candidates
            sims = np.dot(cand_matrix_norm, v_target_norm)
            
            # Mask out the 3 prompt words
            for pw in [w1_q, w2_q, w3_q]:
                if pw in cand_word_to_id:
                    sims[cand_word_to_id[pw]] = -1e9
                    
            top_idx = int(np.argmax(sims))
            pred_word = cand_words[top_idx].lower()
            
            cat_tested += 1
            if pred_word == w4_target:
                cat_correct += 1
                
        acc = (cat_correct / cat_tested) if cat_tested > 0 else 0.0
        results[cat_name] = {'correct': cat_correct, 'total': cat_tested, 'accuracy': acc}
        total_correct += cat_correct
        total_tested += cat_tested
        
    overall_acc = (total_correct / total_tested) if total_tested > 0 else 0.0
    return {'categories': results, 'total_correct': total_correct, 'total_tested': total_tested, 'overall_accuracy': overall_acc}

analogy_results = {}
for mname, model in [('Word2Vec', w2v_model), ('GloVe', glove_model), ('FastText', ft_model)]:
    print(f"\nRunning analogy benchmark on {mname}...")
    t_start = time.time()
    res = evaluate_analogies(model, categories, candidate_limit=30000)
    analogy_results[mname] = res
    print(f"  {mname} Overall Analogy Accuracy: {res['overall_accuracy']*100:.2f}% ({res['total_correct']}/{res['total_tested']}) evaluated in {time.time()-t_start:.2f}s")

# Print summary table for analogies
print("\n" + "-"*65)
print(f"{'Category':<30} | {'Word2Vec':<10} | {'GloVe':<10} | {'FastText':<10}")
print("-"*65)
for cat in categories.keys():
    acc_w2v = analogy_results['Word2Vec']['categories'][cat]['accuracy'] * 100
    acc_glove = analogy_results['GloVe']['categories'][cat]['accuracy'] * 100
    acc_ft = analogy_results['FastText']['categories'][cat]['accuracy'] * 100
    print(f"{cat:<30} | {acc_w2v:>8.2f}% | {acc_glove:>8.2f}% | {acc_ft:>8.2f}%")
print("-"*65)
print(f"{'OVERALL ACCURACY':<30} | {analogy_results['Word2Vec']['overall_accuracy']*100:>8.2f}% | {analogy_results['GloVe']['overall_accuracy']*100:>8.2f}% | {analogy_results['FastText']['overall_accuracy']*100:>8.2f}%")
print("-"*65)

# -------------------------------------------------------------------------
# QUESTION 3: NEURAL NETWORKS (SINGLE HIDDEN LAYER) & COMPARISON
# -------------------------------------------------------------------------
print("\n" + "="*50)
print(" QUESTION 3: SINGLE HIDDEN LAYER NEURAL NETWORKS")
print("="*50)

# We evaluate 6 configurations:
# 3 embeddings (Word2Vec, GloVe, FastText) x 2 representations (Average, TF-IDF weighted)
nn_results = {}

# Hyperparameters: input=300, hidden=128, output=2, epochs=30, batch_size=64, lr=0.08
for mname in ['Word2Vec', 'GloVe', 'FastText']:
    nn_results[mname] = {}
    for rep_type in ['avg', 'tfidf']:
        config_name = f"{mname} ({rep_type.upper()})"
        print(f"\n--- Training Neural Network: {config_name} ---")
        
        X_tr = representations[mname][rep_type]['train']
        X_te = representations[mname][rep_type]['test']
        
        # Design single hidden layer NN
        nn = utils.SingleHiddenLayerNN(input_dim=300, hidden_dim=128, output_dim=2, seed=42)
        
        # Train model
        t_nn = time.time()
        history = nn.fit(X_tr, y_train, X_te, y_test, epochs=30, batch_size=64, lr=0.08, verbose=False)
        train_time = time.time() - t_nn
        
        # Predictions & Macro F1
        test_preds = nn.predict(X_te)
        eval_metrics = utils.compute_macro_f1(y_test, test_preds)
        
        nn_results[mname][rep_type] = {
            'accuracy': eval_metrics['accuracy'],
            'macro_f1': eval_metrics['macro_f1'],
            'confusion_matrix': eval_metrics['confusion_matrix'].tolist(),
            'per_class': eval_metrics['per_class'],
            'history': history,
            'train_time': train_time
        }
        
        print(f"Results for {config_name}:")
        print(f"  Test Accuracy: {eval_metrics['accuracy']*100:.2f}%")
        print(f"  Macro F1-Score: {eval_metrics['macro_f1']:.4f}")
        print(f"  Class 0 (Negative): Prec={eval_metrics['per_class']['class_0']['precision']:.4f}, Rec={eval_metrics['per_class']['class_0']['recall']:.4f}, F1={eval_metrics['per_class']['class_0']['f1']:.4f}")
        print(f"  Class 1 (Positive): Prec={eval_metrics['per_class']['class_1']['precision']:.4f}, Rec={eval_metrics['per_class']['class_1']['recall']:.4f}, F1={eval_metrics['per_class']['class_1']['f1']:.4f}")

# Comparison Table
print("\n" + "="*70)
print(f"{'Embedding Model':<15} | {'Representation':<15} | {'Accuracy':<12} | {'Macro F1':<12}")
print("="*70)
for mname in ['Word2Vec', 'GloVe', 'FastText']:
    for rep_type in ['avg', 'tfidf']:
        acc = nn_results[mname][rep_type]['accuracy'] * 100
        f1 = nn_results[mname][rep_type]['macro_f1']
        rep_label = "Average (a)" if rep_type == 'avg' else "TF-IDF (b)"
        print(f"{mname:<15} | {rep_label:<15} | {acc:>10.2f}% | {f1:>12.4f}")
print("="*70)

# Save all results to json
final_summary = {
    'analogy_results': analogy_results,
    'nn_results': {
        m: {
            r: {
                k: v for k, v in res.items() if k != 'history'
            } for r, res in rep.items()
        } for m, rep in nn_results.items()
    }
}

with open(os.path.join(BASE_DIR, 'experiment_results.json'), 'w') as f:
    json.dump(final_summary, f, indent=2)

# Also save full history for plotting
with open(os.path.join(BASE_DIR, 'nn_training_histories.json'), 'w') as f:
    histories = {
        m: {r: nn_results[m][r]['history'] for r in ['avg', 'tfidf']}
        for m in ['Word2Vec', 'GloVe', 'FastText']
    }
    json.dump(histories, f, indent=2)

print("\nAll experiments successfully completed and saved to experiment_results.json!")

