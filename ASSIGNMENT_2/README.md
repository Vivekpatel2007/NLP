
# ASSIGNMENT 2

[![python](https://img.shields.io/badge/Python-black?style=flat-square&logo=python)](https://www.python.org/) 


This assignment implements a text segmentation using two appraoches : Greedy and Dp Method

Here Segmentation is done for ENGLISH Language.
## Problem / Task

You are given a dataset in JSON format text_segmentation_dataset.json. The dataset is a
snapshot from the large Brown corpus. You need to implement two text segmentation
techniques to segment the text into words and report their performance.

    1. Greedy Based Approach that matches the longest word
    2. Dynamic Programming Approach that increases the log probability of the text [Hint:Frequencies of Words are given]
    
    You need to report two evaluation metrics.
    1. Accuracy
    2. Edit Distance


## Folder Structure

```
ASSIGNMENT_1/
│
├── README.md                        
│
└── Q1.ipynb  # Script for Text Segmentation
│
└──text_segmentation_dataset.json
```
## Approach

The pipeline uses the provided JSON dataset and applies two text segmentation methods:

Greedy Segmentation – Checks possible words from the current position and selects the longest word present in the vocabulary.

Dynamic Programming – Checks different possible word segmentations and selects the one with the highest total log probability, using the given word frequencies.

Evaluation – Both methods are compared with the ground truth using:

Accuracy – Measures correctly predicted words.
Edit Distance – Measures the number of word-level insertions, deletions, and substitutions needed to match the ground truth.### Accuracy

## Results

#### 1. Word Accuracy

    Word Accuracy = Total Correct Words / Total Actual Words

#### 2. Sentence Accuracy

    Sentence Accuracy = Completely Correct Sentences / Total Sentences

#### 3. Average Accuracy per Sentence

    Average Accuracy per Sentence = Sum of Accuracy of All Test Cases / Total Test Cases

### Edit Distance

#### 4. Total Edit Distance

    Total Edit Distance = Sum of Edit Distance of All Test Cases

#### 5. Average Edit Distance

    Average Edit Distance = Total Edit Distance / Total Test Cases

### Summary

| Metric | Greedy | Dynamic Programming |
|---|---:|---:|
| Word Accuracy | 82.21% | **99.10%** |
| Sentence Accuracy | 69.10% | **98.20%** |
| Average Accuracy per Sentence | 83.44% | **99.03%** |
| Total Edit Distance | 1260 | **37** |
| Average Edit Distance | 1.26 | **0.037** |
