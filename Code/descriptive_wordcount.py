import os
import json
import pandas as pd

final = pd.read_csv('df_combined_coref_new.csv')
final.shape
df_combined = final

print(df_combined[df_combined['outlet'] == 'telegraph']['date'].head())

import pandas as pd

def robust_parse_date(date):
    formats = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S%z', '%Y-%m-%d']
    for fmt in formats:
        try:
            return pd.to_datetime(date, format=fmt, errors='raise').date()
        except (ValueError, TypeError):
            continue
    return pd.NaT 

# Converting date column using the robust parser
df_combined['date'] = df_combined['date'].apply(robust_parse_date)

# Checking how many entries were successfully converted and how many were not
print("Converted dates:", df_combined['date'].notna().sum())
print("Unconverted dates (NaT):", df_combined['date'].isna().sum())

# Frequency over time
import spacy
import pandas as pd

# Ensure the SpaCy model is downloaded
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading the SpaCy model...")
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

import pandas as pd
import swifter  # Efficiently apply any function to a pandas dataframe
import spacy

nlp = spacy.load("en_core_web_sm")
df_combined['word_count'] = df_combined['body_resolved'].swifter.apply(
    lambda x: len(nlp(x).text.split()) if x.strip() else 0
)
num_rows_with_zero_words = (df_combined['word_count'] == 0).sum()
df_combined = df_combined[df_combined['word_count'] > 0]
print(f"Number of rows with zero words dropped: {num_rows_with_zero_words}")

df_combined.to_csv('df_combined_coref_words')