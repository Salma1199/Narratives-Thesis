# importing relevant packagers 
import pandas as pd
import os
import json
import re
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Token, Doc
import pandas as pd
import en_core_web_sm
import swifter
from collections import Counter, defaultdict


#Ensure the SpaCy model is downloaded
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading the SpaCy model...")
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

nlp = en_core_web_sm.load()

# Extraction of SVOs adjusted for negations

def extract_SVOs(doc):
    """
    Extract SVO (Subject-Verb-Object) triples using spaCy's dependency parsing
    and lemmatize each part of the SVO, including handling negations.
    """
    svos = []
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            neg = [child for child in token.children if child.dep_ == "neg"]
            verb = f"not {token.lemma_}" if neg else token.lemma_
            subjects = [w for w in token.children if w.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass")]
            objects = [w for w in token.children if w.dep_ in ("dobj", "pobj", "attr", "oprd", "iobj")]
            for subj in subjects:
                for obj in objects:
                    svos.append((subj.lemma_, verb, obj.lemma_))
        # Additional checks for other verb forms
        elif token.pos_ == "VERB":
            neg = [child for child in token.children if child.dep_ == "neg"]
            verb = f"not {token.lemma_}" if neg else token.lemma_
            subjects = [w for w in token.children if w.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass")]
            objects = [w for w in token.children if w.dep_ in ("dobj", "pobj", "attr", "oprd", "iobj")]
            if subjects and objects:
                for subj in subjects:
                    for obj in objects:
                        svos.append((subj.lemma_, verb, obj.lemma_))
    return svos

df_combined = pd.read_csv('df_combined_coref_new.csv')

# Assuming you have already loaded and processed the dataframe sampled_df
df_combined['svos'] = df_combined['body_resolved'].swifter.apply(lambda text: extract_SVOs(nlp(text)))
df_combined['num_svos'] = df_combined['svos'].apply(len)

# Save it externally if needed
df_combined.to_csv('df_combined_svos.csv', index=False)



all_svos_flat = [(svo, df_combined.loc[i, 'outlet']) for i, svos_list in enumerate(df_combined['svos']) for svo in svos_list]
svo_counter = Counter(all_svos_flat)
svo_df = pd.DataFrame(svo_counter.items(), columns=['SVO', 'Total_Count'])
svo_outlets = defaultdict(set)
for svo, outlet in all_svos_flat:
    svo_outlets[svo].add(outlet)

# Create a new list to store the labeled SVOs
labeled_svos = []

# Iterate through all_svos_flat again to label each SVO
for svo, outlet in all_svos_flat:
    if len(svo_outlets[svo]) == 1:
        labeled_svos.append((svo, next(iter(svo_outlets[svo]))))
    else:
        labeled_svos.append((svo, "both"))

# Convert the list to a DataFrame
labeled_svos_df = pd.DataFrame(labeled_svos, columns=['SVO', 'Outlet'])

#Preparing for merging
svo_df_modified = svo_df.copy()
svo_df_modified['SVO'] = svo_df_modified['SVO'].apply(lambda x: x[0])
merged_df = svo_df_modified.merge(labeled_svos_df, on='SVO')
merged_grouped = merged_df.groupby('SVO')['Outlet'].apply(list).reset_index()
merged_grouped['Outlet'] = merged_grouped['Outlet'].apply(lambda x: 'both' if len(set(x)) > 1 else x[0])
final_df = merged_grouped.merge(merged_df.groupby('SVO')['Total_Count'].sum().reset_index(), on='SVO')
final_df

final_df.to_csv('final_df_coref_new.csv', index=False)
