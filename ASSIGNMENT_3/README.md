# ASSIGNMENT 3

[![python](https://img.shields.io/badge/Python-black?style=flat-square\&logo=python)](https://www.python.org/)

This assignment implements a Deterministic Finite Automaton (DFA) and a Finite State Transducer (FST) for Natural Language Processing tasks.

The assignment contains two questions:

1. Constructing a DFA to recognize valid simplified English words.
2. Designing an FST to generate morphological and grammatical features for nouns from the Brown Corpus.

## Problem / Task

### Question 1: DFA for Simplified English Words

Construct a Deterministic Finite Automaton (DFA) that recognizes valid simplified English words.

A valid word:

* Starts with a lowercase letter.
* Is followed by zero or more lowercase letters.

A word must:

* Not start with a digit or punctuation.
* Not contain uppercase letters.
* Not contain spaces, digits, or special characters.

Examples:

```
Accepted: cat, dog, a, zebra

Not Accepted: dog1, 1dog, DogHouse, Dog_house
```

The DFA outputs whether the given string is accepted or not.

---

### Question 2: FST for Morphological Features

You are given all the nouns from the Brown Corpus in `brown_nouns.txt`.

The task is to design a Finite State Transducer (FST) that generates the root word and grammatical features for each noun.

Example:

```
foxes = fox+N+PL
fox = fox+N+SG
```

Where:

```
N  -> Noun
SG -> Singular
PL -> Plural
```

The FST handles the following morphological rules:

| Rule          | Description                                                       | Example                      |
| ------------- | ----------------------------------------------------------------- | ---------------------------- |
| E Insertion   | `e` is added after `s`, `z`, `x`, `ch`, or `sh` before adding `s` | fox → foxes, watch → watches |
| Y Replacement | `y` changes to `ie` before adding `s`                             | try → tries                  |
| S Addition    | `s` is added at the end of the word                               | bag → bags                   |

Incorrect plural forms are identified as:

```
foxs = Invalid Word
watchs = Invalid Word
```

## Folder Structure

```text
ASSIGNMENT_3/
│
├── README.md
│
├── Q1.ipynb              # DFA for simplified English words
│
├── Q2.ipynb              # FST for noun morphological analysis
│
├── brown_nouns.txt       # Nouns extracted from Brown Corpus
│
└── fst_output.txt        # Generated output for processed nouns
```

## Approach

### Q1: Deterministic Finite Automaton

The DFA uses three states:

* Initial State – Checks whether the first character is a lowercase letter.
* Valid State – Continues accepting lowercase letters.
* Dead State – Handles uppercase letters, digits, spaces, punctuation, and special characters.

The automaton accepts the input only when it ends in the valid state.

| Current State | Input         | Next State |
| ------------- | ------------- | ---------- |
| q1            | `a-z`         | q2         |
| q1            | Other         | qd         |
| q2            | `a-z`         | q2         |
| q2            | Other         | qd        |
| qd            | Any character | qd        |


### Q2: Finite State Transducer

The FST processes the input word character by character using state transitions and output functions.

The pipeline is:

1. Load the nouns from `brown_nouns.txt`.
2. Create the input and output alphabets.
3. Define states and their transition tables.
4. Define output functions for every state.
5. Process the input word using the FST.
6. Generate the root word and grammatical features.
7. Validate the generated form using the Brown Corpus noun list.
8. Return `Invalid Word` for incorrect word forms.
9. Process all nouns and save the results to `fst_output.txt`.

#### Transition Table

| Current State | Input | Output | Next State |
|---|---|---|---|
| q1 | `a-z` except `c, s, x, z` | Same character | q1 |
| q1 | `c` | `c` | q2 |
| q1 | `s` | `s` | q6 |
| q1 | `x, z` | Same character | q3 |
| q1 | End of word | `+N+SG` | qend |
| q2 | `h` | `h` | q3 |
| q2 | Other character | Same character | q1 |
| q2 | End of word | `+N+SG` | qend |
| q3 | `e` | `e` | q4 |
| q3 | `s` | `s` | q7 |
| q3 | Other character | Same character | q1 |
| q3 | End of word | `+N+SG` | qend |
| q4 | `s` | `s` | q5 |
| q4 | Other character | Same character | q1 |
| q4 | End of word | `+N+SG` | qend |
| q5 | End of word | `+N+PL` | qend |
| q5 | Other character | Same character | q1 |
| q6 | End of word | `+N+PL` | qend |
| q6 | `e` | `e` | q4 |
| q6 | `h` | `h` | q3 |
| q6 | Other character | Same character | q1 |
| q7 | End of word | `+invalid` | qend |
| q7 | Other character | Same character | q1 |

#### State Meaning

- `q1` – Normal word processing state
- `q2` – Previous character is `c`
- `q3` – Previous characters may form `s`, `x`, `z`, or `ch`
- `q4` – After reading `e`
- `q5` – After reading `es`
- `q6` – Previous character is `s`
- `q7` – Handles invalid plural endings
- `qend` – Final state


## Results

### Q1: DFA

The DFA correctly accepts valid lowercase English words.

Example:

```
cat  -> Accepted
dog  -> Accepted
a    -> Accepted
```

The following invalid inputs are rejected:

```
dog1      -> Not Accepted
1dog      -> Not Accepted
Doghouse  -> Not Accepted
Dog_house -> Not Accepted
```

### Q2: FST

Example outputs:

```
fox          = fox+N+SG
foxes        = fox+N+PL
foxs         = Invalid Word
bag          = bag+N+SG
bags         = bag+N+PL
watch        = watch+N+SG
watches      = watch+N+PL
watchs       = Invalid Word
try          = try+N+SG
tries        = try+N+PL
```

The results for the complete noun corpus are stored in:

```text
fst_output.txt
```
