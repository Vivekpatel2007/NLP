import re
import os
from collections import Counter
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

INPUT_FILE="C:/Users/VIVEK/Desktop/NLP/gu.txt"
URL_PATTERN=r'https?://[^\s]+\.\w{2,}'
EMAIL_PATTERN=r'[\w\.-]+@[\w\.-]+\.\w+'
DATE_PATTERN=r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b'
DECIMAL_PATTERN=r'\d+\.\d+'
NUMBER_PATTERN=r'\d+'
GUJ_WORD=r'[\u0A80-\u0AFF]+'  # Gujarati words range: \u0A80-\u0AFF
PUNCT_PATTERN=r'[^\w\s\u0A80-\u0AFF]'
END=re.compile(r"[.!?]")
special=re.compile(
    f"{URL_PATTERN}|{EMAIL_PATTERN}|{DATE_PATTERN}|{DECIMAL_PATTERN}"
)  # So '.' inside emails/dates/decimals isn't treated as sentence end.
all_special=re.compile(
    f"(?P<url>{URL_PATTERN})"
    f"|(?P<email>{EMAIL_PATTERN})"
    f"|(?P<date>{DATE_PATTERN})"
    f"|(?P<decimal>{DECIMAL_PATTERN})"
    f"|(?P<number>{NUMBER_PATTERN})"
    f"|(?P<gujarati>{GUJ_WORD})"
    f"|(?P<punct>{PUNCT_PATTERN})"
)


def tokenize_sentences(text):
    text=text.strip().replace("\n"," ")

    se=[(m.start(),m.end()) for m in special.finditer(text)]

    sentences=[]
    i=0

    for m in END.finditer(text):
        pos=m.start()

        Is=False
        for s,e in se:
            if s<=pos<e:
                Is=True
                break

        if Is:
            continue

        sentences.append(text[i:pos+1].strip())
        i=pos+1

    if i<len(text):
        sentences.append(text[i:].strip())

    return sentences

def tokenize_words(sentence):
    tokens=[]
    types=[]
    for m in all_special.finditer(sentence):
        tokens.append(m.group())
        types.append(m.lastgroup)
    return tokens, types

def load_all_tokenized_sentences(directory):
    """Load all tokenized sentences from parquet files"""
    all_sentences = []

    for filename in sorted(os.listdir(directory)):
        if filename.startswith("gujarati_tokenized_batch_") and filename.endswith(".parquet"):
            filepath = os.path.join(directory, filename)
            df = pd.read_parquet(filepath)
            all_sentences.extend(df['tokenized_sentence'].tolist())

    return all_sentences

#It is only when you have data downloaded
def local_text_stream(filepath):
    with open(filepath,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield {"text": line}


dataset=local_text_stream(INPUT_FILE)

# When you have not downloaded data locally
# dataset = load_dataset("ai4bharat/IndicCorpV2", "indiccorp_v2", split="guj_Gujr", streaming=True)
def process_and_save_batch(sentences_batch,batch_num):
    rows=[]
    for sent in sentences_batch:

        tokens, token_types=tokenize_words(sent)
        if not tokens:
            continue

        emails=[]
        urls=[]
        dates=[]
        decimals=[]
        integers=[]
        guj_words=[]
        punct=[]
        for i in range(len(tokens)):
            if token_types[i] == "email":
                emails.append(tokens[i])
            elif token_types[i] == "url":
                urls.append(tokens[i])
            elif token_types[i] == "date":
                dates.append(tokens[i])
            elif token_types[i] == "decimal":
                decimals.append(tokens[i])
            elif token_types[i] == "number":
                integers.append(tokens[i])
            elif token_types[i] == "gujarati":
                guj_words.append(tokens[i])
            elif token_types[i] == "punct":
                punct.append(tokens[i])
        punct=list(dict.fromkeys(punct))
        rows.append({
            "original_sentence": sent,
            "tokenized_sentence": " ".join(tokens),
            "tokens": tokens,                       
            "token_types": token_types,              
            "token_count": len(tokens),
            "emails": emails,
            "urls": urls,
            "dates": dates,
            "decimals": decimals,
            "integers": integers,
            "gujarati_words": guj_words,
            "punctuation": punct
        })

    if rows:

        df = pd.DataFrame(rows)
        output_file = os.path.join(
            output_dir,
            f"gujarati_tokenized_batch_{batch_num:04d}.parquet"
        )
        df.to_parquet(
            output_file,
            compression="snappy",
            engine="pyarrow"
        )
        print(f"Saved batch {batch_num} ({len(df)} sentences)")
        return len(df), df["tokenized_sentence"].tolist()

    return 0, []

dataset=local_text_stream(INPUT_FILE)
max_sentences=1000000
batch_size=100000  
output_dir="tokenized_gujarati_corpus"
os.makedirs(output_dir, exist_ok=True)

sentence_count = 0
batch_num = 0
current_batch = []
all_tokens = []
batch_stats = []

for data in dataset:
    if max_sentences is not None and sentence_count >= max_sentences:
        break

    text = data.get("text", "").strip()
    if not text:
        continue

    sentences = tokenize_sentences(text)
    for sent in sentences:
        if max_sentences is not None and sentence_count >= max_sentences:
            break

        current_batch.append(sent)
        sentence_count += 1
        # Process batch when it reaches batch_size
        if len(current_batch) >= batch_size:
            batch_sentences, batch_tokenized = process_and_save_batch(current_batch, batch_num)
            if batch_sentences > 0:
                for tokenized_sent in batch_tokenized:
                    all_tokens.extend(tokenized_sent.split())

                batch_stats.append({
                    'batch_num': batch_num,
                    'sentences': batch_sentences
                })
            current_batch=[]
            batch_num+=1

if current_batch:
    batch_sentences, batch_tokenized = process_and_save_batch(current_batch, batch_num)
    if batch_sentences > 0:
        for tokenized_sent in batch_tokenized:
            all_tokens.extend(tokenized_sent.split())

        batch_stats.append({
            'batch_num': batch_num,
            'sentences': batch_sentences
        })

total_sentences=sum(stat['sentences'] for stat in batch_stats)
total_words=len(all_tokens)
total_characters=sum(len(token) for token in all_tokens)
unique_tokens=set(all_tokens)
avg_sentence_length=total_words / total_sentences if total_sentences else 0
avg_word_length = total_characters / total_words if total_words else 0
type_token_ratio = len(unique_tokens) / total_words if total_words else 0

print(" FINAL CORPUS STATISTICS")
print(f" Total number of sentences         : {total_sentences:,}")
print(f" Total number of words             : {total_words:,}")
print(f" Total number of characters        : {total_characters:,}")
print(f" Average sentence length (words)   : {avg_sentence_length:.2f}")
print(f" Average word length (characters)  : {avg_word_length:.2f}")
print(f" Type/Token Ratio (TTR)            : {type_token_ratio:.4f}")
print(f" Number of batches created         : {len(batch_stats)}")

token_counts = Counter(all_tokens)
total_unique_tokens = len(token_counts)

stats_data = {
    'metric': [
        'Total Sentences', 'Total Words', 'Total Characters',
        'Average Sentence Length', 'Average Word Length', 'Type-Token Ratio',
        'Unique Tokens'
    ],
    'value': [
        total_sentences, total_words, total_characters,
        avg_sentence_length, avg_word_length, type_token_ratio,
        total_unique_tokens
    ]
}

stats_df = pd.DataFrame(stats_data)
stats_file = os.path.join(output_dir, "corpus_statistics.parquet")
stats_df.to_parquet(stats_file, compression='snappy')

print(f"\n Files saved in directory: {output_dir}/")
print("   - gujarati_tokenized_batch_XXXX.parquet (tokenized sentences)")
print("   - corpus_statistics.parquet (overall statistics)")

all_tokenized=load_all_tokenized_sentences(output_dir)
print(f"Total loaded sentences: {len(all_tokenized)}")