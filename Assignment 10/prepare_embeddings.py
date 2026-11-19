"""
prepare_embeddings.py - Convert GloVe and FastText to memory-mapped KeyedVectors for instant loading.
"""

import os
import time
from gensim.models import KeyedVectors

base_dir = os.path.dirname(os.path.abspath(__file__))

def prepare_glove():
    kv_path = os.path.join(base_dir, 'glove.6B.300d.kv')
    txt_path = os.path.join(base_dir, 'glove.6B.300d.txt')
    if os.path.exists(kv_path):
        print(f"GloVe KeyedVectors already exists at {kv_path}")
        return
    print("Converting GloVe 300d to KeyedVectors format...")
    t0 = time.time()
    # GloVe does not have header, load with no_header=True
    model = KeyedVectors.load_word2vec_format(txt_path, binary=False, no_header=True)
    model.save(kv_path)
    print(f"GloVe saved to {kv_path} in {time.time()-t0:.2f}s ({len(model.key_to_index)} words)")

def prepare_fasttext():
    kv_path = os.path.join(base_dir, 'wiki-news-300d-1M-subword.kv')
    vec_path = os.path.join(base_dir, 'wiki-news-300d-1M-subword.vec')
    if os.path.exists(kv_path):
        print(f"FastText KeyedVectors already exists at {kv_path}")
        return
    print("Converting FastText 300d to KeyedVectors format...")
    t0 = time.time()
    # FastText has header: 1000000 300
    model = KeyedVectors.load_word2vec_format(vec_path, binary=False)
    model.save(kv_path)
    print(f"FastText saved to {kv_path} in {time.time()-t0:.2f}s ({len(model.key_to_index)} words)")

if __name__ == '__main__':
    prepare_glove()
    prepare_fasttext()

