# Hindi Part-of-Speech (POS) Tagger

A comprehensive statistical and sequential Part-of-Speech (POS) tagging system for Hindi, implementing **Naive Bayes** and **Hidden Markov Models (HMM)** with **Forward Viterbi**, **Backward Viterbi**, and **Forward-Backward Viterbi** decoding algorithms.

Developed for NLP Lab Assignment 7 (2026).

---

## 📋 Table of Contents
1. [Overview & Assignment Requirements](#overview--assignment-requirements)
2. [Dataset & Annotation Scheme](#dataset--annotation-scheme)
3. [Algorithm Formulations & Architectures](#algorithm-formulations--architectures)
   - [1. Naive Bayes Classifier](#1-naive-bayes-classifier)
   - [2. Hidden Markov Model (HMM)](#2-hidden-markov-model-hmm)
   - [2a. Standard Forward Viterbi](#2a-standard-forward-viterbi)
   - [2b. Backward Viterbi](#2b-backward-viterbi)
   - [2c. Forward-Backward Viterbi](#2c-forward-backward-viterbi)
4. [Experimental Results & Evaluation](#experimental-results--evaluation)
   - [Overall Model Comparison](#overall-model-comparison)
   - [Per-Tag F1-Score Breakdown](#per-tag-f1-score-breakdown)
5. [Generated Visualizations & Plots](#generated-visualizations--plots)
6. [Comparative Analysis & Key Insights](#comparative-analysis--key-insights)
7. [Directory Structure](#directory-structure)
8. [How to Run](#how-to-run)

---

## 1. Overview & Assignment Requirements

The assignment specifies the following tasks:
1. **Naive Bayes Classifier**: Predict the POS tag of each word using:
   - **a. Length of word**: Binary feature (`"MORE"` if $\text{len}(\text{word}) > 4$ else `"LESS"`)
   - **b. Actual Word** ($w_i$)
   - **c. Previous Word** ($w_{i-1}$)
   - **d. Next Word** ($w_{i+1}$)
   - **e. Previous Tag** ($t_{i-1}$)
2. **Hidden Markov Model (HMM)**: Predict the tag sequence given a sentence:
   - **a. Standard Forward Viterbi**
   - **b. Backward Viterbi**
   - **c. Forward-Backward Viterbi**
3. **Probability Estimation & Evaluation**:
   - Compute all probabilities from the training set (`hindi-train-pos.txt`).
   - Evaluate on both development (`hindi-dev-pos.txt`) and test (`hindi-test-pos.txt`) splits.
   - Report per-tag F1-scores, compare all algorithms, and plot comparative charts.

---

## 2. Dataset & Annotation Scheme

The dataset uses the standardized **Bureau of Indian Standards (BIS) / ILMT 24-tag POS tagset** for Hindi.

### Dataset Statistics
| Split | Sentences | Tokens | Vocab Size | OOV Tokens | OOV Rate |
|---|---|---|---|---|---|
| **Train** | 15,087 | 325,710 | 18,279 | — | — |
| **Dev** | 1,866 | 40,762 | 6,564 | 1,707 | 4.19% |
| **Test** | 1,910 | 41,415 | 6,654 | 1,899 | 4.59% |

### Complete 24-Tag Scheme Description
| Tag | Description in Hindi | Grammatical Category |
|---|---|---|
| `N_NN` | जातिवाचक संज्ञा | Common Noun |
| `PSP` | परसर्ग / कारक | Postposition |
| `N_NNP` | व्यक्तिवाचक संज्ञा | Proper Noun |
| `V_VM` | मुख्य क्रिया | Main Verb |
| `V_VAUX` | सहायक क्रिया | Auxiliary Verb |
| `RD_PUNC` | विराम चिह्न | Punctuation |
| `JJ` | विशेषण | Adjective |
| `PR_PRP` | व्यक्तिवाचक सर्वनाम | Personal Pronoun |
| `QT_QTC` | गणनावाचक परिमाण | Cardinal Quantifier |
| `CC_CCD` | समानाधिकरण समुच्चयबोधक | Coordinating Conjunction |
| `CC_CCS` | व्याधिकरण समुच्चयबोधक | Subordinating Conjunction |
| `N_NST` | दिक्कालवाचक संज्ञा | Spatial/Temporal Noun |
| `RP_RPD` | निपात | Default Particle |
| `DM_DMD` | निर्देशात्मक सर्वनाम | Deictic Demonstrative |
| `RD_SYM` | प्रतीक / चिह्न | Symbol |
| `QT_QTF` | सामान्य परिमाणवाचक | General Quantifier |
| `RP_NEG` | नकारार्थक निपात | Negative Particle |
| `RB` | क्रिया विशेषण | Adverb |
| `PR_PRF` | निजवाचक सर्वनाम | Reflexive Pronoun |
| `PR_PRL` | संबंधवाचक सर्वनाम | Relative Pronoun |
| `QT_QTO` | क्रमवाचक परिमाण | Ordinal Quantifier |
| `RP_INTF` | तीव्रतादर्शक निपात | Intensifier Particle |
| `PR_PRQ` | प्रश्नवाचक सर्वनाम | Question Pronoun |
| `RP_INJ` | विस्मयादिबोधक निपात | Interjection Particle |

---

## 3. Algorithm Formulations & Architectures

### 1. Naive Bayes Classifier
For word position $i$ with feature vector $X_i = (f_{\text{len}}, w_i, w_{i-1}, w_{i+1}, t_{i-1})$:
$$\hat{t}_i = \arg\max_{t \in \mathcal{T}} \left[ \log P(t) + \log P(f_{\text{len}} \mid t) + \log P(w_i \mid t) + \log P(w_{i-1} \mid t) + \log P(w_{i+1} \mid t) + \log P(t_{i-1} \mid t) \right]$$

Laplace / Lidstone Add-$\alpha$ smoothing ($\alpha = 0.1$):
$$P(f_k = v \mid t) = \frac{C(t, f_k = v) + \alpha}{C(t) + \alpha \cdot (|V_{f_k}| + 1)}$$

- **Greedy Sequential Inference**: $\hat{t}_i$ is predicted using the previously predicted tag $\hat{t}_{i-1}$ as the feature.
- **Teacher-Forced Inference**: Uses the ground-truth previous tag for comparative theoretical upper-bound evaluation.

---

### 2. Hidden Markov Model (HMM)
The HMM models the joint sequence probability:
$$P(W, T) = P(t_1) \prod_{i=2}^N P(t_i \mid t_{i-1}) \prod_{i=1}^N P(w_i \mid t_i)$$

Parameters:
- Initial log-probabilities: $\log \pi_j = \log \frac{C(\text{<BOS>}, t_j) + \alpha_{\text{init}}}{N_{\text{sents}} + \alpha_{\text{init}} K}$
- Transition log-probabilities: $\log A_{ij} = \log \frac{C(t_i, t_j) + \alpha_{\text{trans}}}{C(t_i) + \alpha_{\text{trans}} K}$
- Emission log-probabilities: $\log B_j(w) = \log \frac{C(t_j, w) + \beta_{\text{emit}}}{C(t_j) + \beta_{\text{emit}} (|V| + 1)}$
- OOV word emission: $\log B_j(\text{UNK}) = \log \frac{\beta_{\text{emit}}}{C(t_j) + \beta_{\text{emit}} (|V| + 1)}$

#### 2a. Standard Forward Viterbi (Left-to-Right)
- **Initialization ($t = 0$):**
  $$V_0(j) = \log \pi_j + \log B_j(w_0)$$
- **Forward Recursion ($t = 1 \dots N-1$):**
  $$V_t(j) = \max_{i \in \mathcal{T}} \left[ V_{t-1}(i) + \log A_{ij} \right] + \log B_j(w_t)$$
  $$bp_t(j) = \arg\max_{i \in \mathcal{T}} \left[ V_{t-1}(i) + \log A_{ij} \right]$$
- **Termination & Backtracking:**
  $$\hat{t}_{N-1} = \arg\max_{j} V_{N-1}(j), \quad \hat{t}_{t-1} = bp_t(\hat{t}_t)$$

#### 2b. Backward Viterbi (Right-to-Left)
- **Base Case ($t = N-1$):**
  $$V^B_{N-1}(j) = 0.0$$
- **Backward Recursion ($t = N-2 \dots 0$):**
  $$V^B_t(j) = \max_{k \in \mathcal{T}} \left[ \log A_{jk} + \log B_k(w_{t+1}) + V^B_{t+1}(k) \right]$$
  $$fp_t(j) = \arg\max_{k \in \mathcal{T}} \left[ \log A_{jk} + \log B_k(w_{t+1}) + V^B_{t+1}(k) \right]$$
- **Termination & Forward-Tracking:**
  $$\hat{t}_0 = \arg\max_{j} \left[ \log \pi_j + \log B_j(w_0) + V^B_0(j) \right], \quad \hat{t}_{t+1} = fp_t(\hat{t}_t)$$

#### 2c. Forward-Backward Viterbi
1. **Bidirectional Viterbi (Trellis Fusion)**:
   Computes both $V^F$ and $V^B$. At the midpoint $m = \lfloor N/2 \rfloor$:
   $$\hat{t}_m = \arg\max_{j} \left[ V^F_m(j) + V^B_m(j) \right]$$
   Then traces backwards to $0$ using $bp$ and forwards to $N-1$ using $fp$.
2. **Posterior Marginal Decoding (Sum-Product Forward-Backward)**:
   Computes forward variables $\log \alpha$ and backward variables $\log \beta$ using `logsumexp`:
   $$\log \gamma_t(j) = \log \alpha_t(j) + \log \beta_t(j)$$
   $$\hat{t}_t = \arg\max_{j} \gamma_t(j)$$

---

## 4. Experimental Results & Evaluation

### Overall Model Comparison

| Model | Dev Acc (%) | Dev Macro-F1 (%) | Dev W-F1 (%) | Test Acc (%) | Test Macro-F1 (%) | Test W-F1 (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Naive Bayes (Greedy)** | 92.47% | 83.31% | 92.49% | 92.57% | 84.44% | 92.59% |
| **Naive Bayes (Gold PrevTag)** | 93.10% | 83.86% | 93.13% | 93.15% | 84.70% | 93.17% |
| **HMM Forward Viterbi** | 92.99% | 82.78% | 93.09% | 92.90% | 83.26% | 93.01% |
| **HMM Backward Viterbi** | 92.99% | 82.78% | 93.09% | 92.90% | 83.26% | 93.01% |
| **HMM Forward-Backward Viterbi** | 92.99% | 82.78% | 93.09% | 92.90% | 83.26% | 93.01% |
| **HMM Posterior Decoding** | **93.12%** | **83.79%** | **93.18%** | **92.93%** | **83.95%** | **93.00%** |

---

### Per-Tag F1-Score Breakdown (Test Set)

| POS Tag | Linguistic Category | Support | NB (Greedy) | HMM Fwd Vit | HMM Bwd Vit | HMM F-B Vit | HMM Post Dec |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `N_NN` | Common Noun | 9,303 | 0.901 | 0.911 | 0.911 | 0.911 | 0.909 |
| `PSP` | Postposition | 8,245 | 0.982 | 0.986 | 0.986 | 0.986 | 0.985 |
| `N_NNP` | Proper Noun | 5,296 | 0.841 | 0.840 | 0.840 | 0.840 | 0.837 |
| `V_VM` | Main Verb | 4,174 | 0.963 | 0.953 | 0.953 | 0.953 | 0.956 |
| `RD_PUNC` | Punctuation | 2,549 | 0.994 | 0.996 | 0.996 | 0.996 | 0.996 |
| `V_VAUX` | Auxiliary Verb | 2,534 | 0.976 | 0.967 | 0.967 | 0.967 | 0.969 |
| `JJ` | Adjective | 2,258 | 0.887 | 0.905 | 0.905 | 0.905 | 0.906 |
| `PR_PRP` | Personal Pronoun | 1,338 | 0.907 | 0.962 | 0.962 | 0.962 | 0.958 |
| `QT_QTC` | Cardinal Quantifier | 874 | 0.920 | 0.871 | 0.871 | 0.871 | 0.879 |
| `CC_CCD` | Coordinating Conjunction | 772 | 0.962 | 0.972 | 0.972 | 0.972 | 0.974 |
| `CC_CCS` | Subordinating Conjunction | 736 | 0.967 | 0.970 | 0.970 | 0.970 | 0.977 |
| `N_NST` | Spatial/Temporal Noun | 625 | 0.889 | 0.932 | 0.932 | 0.932 | 0.920 |
| `RP_RPD` | Default Particle | 617 | 0.883 | 0.926 | 0.926 | 0.926 | 0.921 |
| `DM_DMD` | Deictic Demonstrative | 516 | 0.898 | 0.927 | 0.927 | 0.927 | 0.927 |
| `RD_SYM` | Symbol | 464 | 0.943 | 0.881 | 0.881 | 0.881 | 0.880 |
| `QT_QTF` | General Quantifier | 323 | 0.856 | 0.923 | 0.923 | 0.923 | 0.925 |
| `RP_NEG` | Negative Particle | 243 | 0.888 | 0.849 | 0.849 | 0.849 | 0.880 |
| `RB` | Adverb | 167 | 0.661 | 0.807 | 0.807 | 0.807 | 0.792 |
| `PR_PRF` | Reflexive Pronoun | 151 | 0.903 | 0.680 | 0.680 | 0.680 | 0.739 |
| `PR_PRL` | Relative Pronoun | 91 | 0.725 | 0.677 | 0.677 | 0.677 | 0.690 |
| `QT_QTO` | Ordinal Quantifier | 78 | 0.667 | 0.767 | 0.767 | 0.767 | 0.798 |
| `PR_PRQ` | Question Pronoun | 31 | 0.830 | 0.765 | 0.765 | 0.765 | 0.805 |
| `RP_INTF` | Intensifier Particle | 29 | 0.821 | 0.514 | 0.514 | 0.514 | 0.524 |
| `RP_INJ` | Interjection Particle | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

---

## 5. Generated Visualizations & Plots

All plots are generated in high-resolution (300 DPI) and stored in `plots/`:
1. **`plots/per_tag_f1_comparison_dev.png`**: Per-tag F1 bar chart across all algorithms on Dev set.
2. **`plots/per_tag_f1_comparison_test.png`**: Per-tag F1 bar chart across all algorithms on Test set.
3. **`plots/per_tag_f1_heatmap_test.png`**: Comparative heatmap of F1-scores across all 24 tags and 5 models.
4. **`plots/overall_metrics_comparison.png`**: Overall Accuracy, Macro F1, and Weighted F1 comparison on Dev vs Test.
5. **`plots/confusion_matrix_hmm_forward_viterbi.png`**: Normalized confusion matrix for HMM Forward Viterbi on Test set.
6. **`plots/confusion_matrix_naive_bayes.png`**: Normalized confusion matrix for Naive Bayes on Test set.
7. **`plots/tag_frequency_vs_f1_test.png`**: Scatter plot analyzing the correlation between tag support frequency and prediction F1 score.

---

## 6. Comparative Analysis & Key Insights

1. **Exact Equivalence of Viterbi Directional Implementations**:
   - Forward Viterbi, Backward Viterbi, and Bidirectional Forward-Backward Viterbi produce **identical sequence predictions** (Accuracy = 92.9011%, Weighted F1 = 93.0075% on Test; 92.9910% on Dev).
   - This mathematically verifies that dynamic programming correctly finds the global mode $\arg\max_T P(W, T)$ irrespective of recursion direction.

2. **Posterior Marginal Decoding Advantages**:
   - Posterior Decoding achieves the highest test macro-F1 (83.95%) and dev accuracy (93.12%) because it minimizes pointwise classification risk (word error rate) rather than entire sequence error rate.

3. **Strengths of Naive Bayes with 5 Features**:
   - Despite ignoring full sentence sequence transitions, Naive Bayes achieves **92.57% test accuracy**.
   - Contextual word features ($w_{i-1}, w_{i+1}$) provide valuable disambiguation for intensifiers (`RP_INTF`, F1 = 0.821 vs HMM's 0.514) and reflexive pronouns (`PR_PRF`, F1 = 0.903 vs HMM's 0.680), where word neighborhood matters more than state bigrams.
   - The binary word length feature ($\text{len} > 4$) strongly separates functional closed-class categories from open-class content words.

4. **Major Error Modes in Hindi POS Tagging**:
   - **`N_NN` vs `N_NNP`**: The lack of capitalization in Devanagari script makes proper vs common noun differentiation challenging, primarily relying on semantic context and preceding titles.
   - **`JJ` vs `N_NN`**: Hindi frequently uses nouns as attributive modifiers in noun-noun compounds.

---

## 7. Directory Structure

```
Hindi_POS_Tagger/
├── data/
│   ├── hindi-train-pos.txt             # Training dataset (325,710 tokens)
│   ├── hindi-dev-pos.txt               # Development dataset (40,762 tokens)
│   └── hindi-test-pos.txt              # Test dataset (41,415 tokens)
├── src/
│   ├── __init__.py                     # Package init
│   ├── data_loader.py                  # Data loading & dataset statistics
│   ├── naive_bayes.py                  # Naive Bayes Classifier
│   ├── hmm.py                          # HMM with Forward, Backward, F-B Viterbi
│   ├── evaluator.py                    # Metric calculation & reports
│   └── visualizer.py                   # Chart generation (Matplotlib/Seaborn)
├── plots/
│   ├── per_tag_f1_comparison_dev.png
│   ├── per_tag_f1_comparison_test.png
│   ├── per_tag_f1_heatmap_test.png
│   ├── overall_metrics_comparison.png
│   ├── confusion_matrix_hmm_forward_viterbi.png
│   ├── confusion_matrix_naive_bayes.png
│   └── tag_frequency_vs_f1_test.png
├── results/
│   ├── dev_overall_summary.csv
│   ├── test_overall_summary.csv
│   ├── dev_per_tag_f1_scores.csv
│   ├── test_per_tag_f1_scores.csv
│   └── detailed_classification_reports.txt
├── Hindi_POS_Tagger_Walkthrough.ipynb  # Executed Jupyter Notebook walkthrough
├── main.py                             # Main experiment execution script
└── README.md                           # Documentation & lab report
```

---

## 8. How to Run

### Run Full Pipeline
To run training, evaluation on Dev & Test, generate all comparison CSV tables and plots:
```bash
python main.py
```

### Launch Jupyter Notebook
To interactively view the notebook with pre-executed code cells, formatted output tables, and visualizations:
```bash
jupyter notebook Hindi_POS_Tagger_Walkthrough.ipynb
```
