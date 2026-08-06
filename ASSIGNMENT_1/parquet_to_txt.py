import pandas as pd
df = pd.read_parquet('ASSIGNMENT_1/tokenized_gujarati_corpus/gujarati_tokenized_batch_0000.parquet') # Output of Tokenizer.py
df.to_csv('ASSIGNMENT_1/out.csv', index=False)             
df.to_csv('ASSIGNMENT_1/out.txt', index=False, sep='\t')  

df.to_excel('output.xlsx', index=False) # Save the DataFrame to an Excel (XLSX) file
