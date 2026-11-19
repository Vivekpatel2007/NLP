"""
generate_plots_and_tables.py - Generates comparison plots and CSV reports for Assignment 10.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load results
with open(os.path.join(BASE_DIR, 'experiment_results.json'), 'r') as f:
    res = json.load(f)

with open(os.path.join(BASE_DIR, 'nn_training_histories.json'), 'r') as f:
    histories = json.load(f)

analogy_res = res['analogy_results']
nn_res = res['nn_results']

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# -------------------------------------------------------------
# PLOT 1: Analogy Accuracy Comparison by Category & Overall
# -------------------------------------------------------------
categories = list(analogy_res['Word2Vec']['categories'].keys())
w2v_accs = [analogy_res['Word2Vec']['categories'][c]['accuracy'] * 100 for c in categories]
glove_accs = [analogy_res['GloVe']['categories'][c]['accuracy'] * 100 for c in categories]
ft_accs = [analogy_res['FastText']['categories'][c]['accuracy'] * 100 for c in categories]

# Add overall
categories_with_overall = categories + ['OVERALL']
w2v_accs.append(analogy_res['Word2Vec']['overall_accuracy'] * 100)
glove_accs.append(analogy_res['GloVe']['overall_accuracy'] * 100)
ft_accs.append(analogy_res['FastText']['overall_accuracy'] * 100)

y = np.arange(len(categories_with_overall))
height = 0.28

fig, ax = plt.subplots(figsize=(12, 10))
ax.barh(y - height, w2v_accs, height, label='Word2Vec (GoogleNews-300)', color='#1f77b4')
ax.barh(y, glove_accs, height, label='GloVe (6B-300d)', color='#ff7f0e')
ax.barh(y + height, ft_accs, height, label='FastText (wiki-news-300d)', color='#2ca02c')

ax.set_yticks(y)
ax.set_yticklabels(categories_with_overall, fontsize=11)
ax.invert_yaxis()
ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Q2: Word Analogy Accuracy by Category on questions-words.txt', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11, loc='lower right')
ax.set_xlim(0, 105)

# Highlight OVERALL
for i, v in enumerate(w2v_accs):
    if i == len(w2v_accs) - 1:
        ax.get_yticklabels()[i].set_fontweight('bold')
        ax.get_yticklabels()[i].set_color('#8B0000')

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'analogy_accuracy_comparison.png'), dpi=300)
plt.close()
print("Saved analogy_accuracy_comparison.png")

# -------------------------------------------------------------
# PLOT 2: Macro F1-Score Comparison for Q3
# -------------------------------------------------------------
models = ['Word2Vec', 'GloVe', 'FastText']
avg_f1s = [nn_res[m]['avg']['macro_f1'] for m in models]
tfidf_f1s = [nn_res[m]['tfidf']['macro_f1'] for m in models]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 6))
rects1 = ax.bar(x - width/2, avg_f1s, width, label='Way a: Average Vector', color='#4C72B0')
rects2 = ax.bar(x + width/2, tfidf_f1s, width, label='Way b: TF-IDF Weighted Vector', color='#55A868')

ax.set_ylabel('Macro F1-Score', fontsize=12, fontweight='bold')
ax.set_title('Q3: Single Hidden Layer NN Macro F1-Score Comparison', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.0)
ax.legend(fontsize=11)

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'{h:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'macro_f1_comparison.png'), dpi=300)
plt.close()
print("Saved macro_f1_comparison.png")

# -------------------------------------------------------------
# PLOT 3: Training Loss Curves
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

colors = {'avg': '#1f77b4', 'tfidf': '#2ca02c'}
for i, m in enumerate(models):
    ax = axes[i]
    for rep in ['avg', 'tfidf']:
        tr_loss = histories[m][rep]['train_loss']
        val_loss = histories[m][rep]['val_loss']
        epochs = range(1, len(tr_loss) + 1)
        lbl = 'Average' if rep == 'avg' else 'TF-IDF'
        ax.plot(epochs, tr_loss, label=f'{lbl} (Train)', color=colors[rep], linestyle='-')
        ax.plot(epochs, val_loss, label=f'{lbl} (Val)', color=colors[rep], linestyle='--')
        
    ax.set_title(f'{m} Embeddings', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=11)
    if i == 0:
        ax.set_ylabel('Cross-Entropy Loss', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Q3: Neural Network Training and Validation Loss Curves (30 Epochs)', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'training_loss_curves.png'), dpi=300)
plt.close()
print("Saved training_loss_curves.png")

# -------------------------------------------------------------
# PLOT 4: Confusion Matrices
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(14, 9))

for row_idx, rep in enumerate(['avg', 'tfidf']):
    rep_title = "Average Vector" if rep == 'avg' else "TF-IDF Weighted"
    for col_idx, m in enumerate(models):
        ax = axes[row_idx, col_idx]
        cm = np.array(nn_res[m][rep]['confusion_matrix'])
        f1 = nn_res[m][rep]['macro_f1']
        
        im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
        for r in range(2):
            for c in range(2):
                ax.text(c, r, f"{cm[r, c]}", ha="center", va="center",
                        color="white" if cm[r, c] > cm.max()/2 else "black", fontsize=12, fontweight='bold')
                
        ax.set_title(f"{m} - {rep_title}\nMacro F1: {f1:.4f}", fontsize=11, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred Neg (1)', 'Pred Pos (2)'])
        ax.set_yticklabels(['True Neg (1)', 'True Pos (2)'])

plt.suptitle('Q3: Test Confusion Matrices (1000 Test Samples)', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'confusion_matrices.png'), dpi=300)
plt.close()
print("Saved confusion_matrices.png")

# -------------------------------------------------------------
# SAVE CSV SUMMARY TABLES
# -------------------------------------------------------------
# Analogy summary CSV
with open(os.path.join(BASE_DIR, 'analogy_category_results.csv'), 'w') as f:
    f.write("Category,Word2Vec_Accuracy,GloVe_Accuracy,FastText_Accuracy,Word2Vec_Correct,GloVe_Correct,FastText_Correct,Total_Questions\n")
    for cat in categories:
        w_acc = analogy_res['Word2Vec']['categories'][cat]['accuracy']
        g_acc = analogy_res['GloVe']['categories'][cat]['accuracy']
        f_acc = analogy_res['FastText']['categories'][cat]['accuracy']
        w_c = analogy_res['Word2Vec']['categories'][cat]['correct']
        g_c = analogy_res['GloVe']['categories'][cat]['correct']
        f_c = analogy_res['FastText']['categories'][cat]['correct']
        tot = analogy_res['Word2Vec']['categories'][cat]['total']
        f.write(f'"{cat}",{w_acc:.4f},{g_acc:.4f},{f_acc:.4f},{w_c},{g_c},{f_c},{tot}\n')
    # Overall
    ow_acc = analogy_res['Word2Vec']['overall_accuracy']
    og_acc = analogy_res['GloVe']['overall_accuracy']
    of_acc = analogy_res['FastText']['overall_accuracy']
    ow_c = analogy_res['Word2Vec']['total_correct']
    og_c = analogy_res['GloVe']['total_correct']
    of_c = analogy_res['FastText']['total_correct']
    otot = analogy_res['Word2Vec']['total_tested']
    f.write(f'"OVERALL",{ow_acc:.4f},{og_acc:.4f},{of_acc:.4f},{ow_c},{og_c},{of_c},{otot}\n')

print("Saved analogy_category_results.csv")

# Q3 Neural Network summary CSV
with open(os.path.join(BASE_DIR, 'neural_network_f1_results.csv'), 'w') as f:
    f.write("Embedding_Model,Representation_Method,Accuracy,Macro_F1,Class0_F1,Class1_F1,Class0_Precision,Class0_Recall,Class1_Precision,Class1_Recall\n")
    for m in models:
        for rep in ['avg', 'tfidf']:
            d = nn_res[m][rep]
            rep_name = "Average" if rep == 'avg' else "TF-IDF Weighted"
            c0 = d['per_class']['class_0']
            c1 = d['per_class']['class_1']
            f.write(f"{m},{rep_name},{d['accuracy']:.4f},{d['macro_f1']:.4f},{c0['f1']:.4f},{c1['f1']:.4f},{c0['precision']:.4f},{c0['recall']:.4f},{c1['precision']:.4f},{c1['recall']:.4f}\n")

print("Saved neural_network_f1_results.csv")
print("All plots and CSV reports generated successfully!")

