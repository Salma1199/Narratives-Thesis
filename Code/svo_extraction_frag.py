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

df_combined = pd.read_csv('df_combined_coref_new.csv')# Assuming you have already loaded and processed the dataframe sampled_df

df_combined['svos'] = df_combined['body_resolved'].swifter.apply(lambda text: extract_SVOs(nlp(text)))
df_combined['num_svos'] = df_combined['svos'].apply(len)

eu_countries = {
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Estonia",
    "Finland", "France", "Germany", "Greece", "Hungary", "Ireland", "Italy", "Latvia", "Lithuania",
    "Luxembourg", "Malta", "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia",
    "Spain", "Sweden"
}

non_eu_countries = {
    "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cape Verde", "Cameroon", 
    "Central African Republic", "Chad", "Comoros", "Congo", "Djibouti", "Egypt", "Equatorial Guinea", 
    "Eritrea", "Eswatini", "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Ivory Coast", 
    "Kenya", "Lesotho", "Liberia", "Libya", "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", "Morocco", 
    "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda", "Sao Tome and Principe", "Senegal", "Seychelles", 
    "Sierra Leone", "Somalia", "South Africa", "South Sudan", "Sudan", "Tanzania", "Togo", "Tunisia", "Uganda", 
    "Zambia", "Zimbabwe", "Antigua and Barbuda", "Argentina", "Bahamas", "Barbados", "Belize", "Bolivia", "Brazil", 
    "Canada", "Chile", "Colombia", "Costa Rica", "Cuba", "Dominica", "Dominican Republic", "Ecuador", "El Salvador", 
    "Grenada", "Guatemala", "Guyana", "Haiti", "Honduras", "Jamaica", "Mexico", "Nicaragua", "Panama", "Paraguay", 
    "Peru", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Suriname", "Trinidad and Tobago", 
    "United States", "Uruguay", "Venezuela", "Afghanistan", "Armenia", "Azerbaijan", "Bahrain", "Bangladesh", "Bhutan", 
    "Brunei", "Cambodia", "China", "Georgia", "India", "Indonesia", "Iran", "Iraq", "Israel", "Japan", "Jordan", 
    "Kazakhstan", "Kuwait", "Kyrgyzstan", "Laos", "Lebanon", "Malaysia", "Maldives", "Mongolia", "Myanmar", "Nepal", 
    "North Korea", "Oman", "Pakistan", "Palestine", "Philippines", "Qatar", "Saudi Arabia", "Singapore", "South Korea", 
    "Sri Lanka", "Syria", "Taiwan", "Tajikistan", "Thailand", "Timor-Leste", "Turkmenistan", "United Arab Emirates", 
    "Uzbekistan", "Vietnam", "Yemen", "Australia", "Fiji", "Kiribati", "Marshall Islands", "Micronesia", "Nauru", 
    "New Zealand", "Palau", "Papua New Guinea", "Samoa", "Solomon Islands", "Tonga", "Tuvalu", "Vanuatu", "Albania", 
    "Andorra", "Belarus", "Bosnia and Herzegovina", "Iceland", "Kosovo", "Liechtenstein", "Moldova", "Monaco", 
    "Montenegro", "North Macedonia", "Norway", "Russia", "San Marino", "Serbia", "Switzerland", "Turkey", "Ukraine", 
    "Vatican City"
}

global_north_countries = {
    "United States", "Canada", "Australia", "New Zealand", "Japan", "South Korea",
    "Singapore", "Israel", "Norway", "Switzerland", "Iceland", "Liechtenstein",
    "United Kingdom", "Germany", "France", "Italy", "Spain", "Portugal", "Sweden", 
    "Denmark", "Finland", "Austria", "Belgium", "Luxembourg", "Netherlands", 
    "Ireland", "Greece", "Cyprus", "Malta", "Czech Republic", "Slovakia",
    "Slovenia", "Estonia", "Latvia", "Lithuania", "Poland", "Hungary", "Croatia",
    "Romania", "Bulgaria"}

global_south_countries = {
    "Mexico", "Guatemala", "Belize", "El Salvador", "Honduras", "Nicaragua", "Costa Rica", "Panama",
    "Colombia", "Venezuela", "Guyana", "Suriname", "Ecuador", "Peru", "Bolivia", "Brazil", "Paraguay", 
    "Chile", "Argentina", "Uruguay",
    "Morocco", "Algeria", "Tunisia", "Libya", "Egypt", "Sudan", "South Sudan", "Western Sahara", 
    "Mauritania", "Mali", "Niger", "Chad", "Senegal", "Gambia", "Guinea-Bissau", "Guinea", "Sierra Leone",
    "Liberia", "Côte d'Ivoire", "Burkina Faso", "Ghana", "Togo", "Benin", "Nigeria", "Cameroon", "Cape Verde",
    "Sao Tome and Principe", "Equatorial Guinea", "Gabon", "Central African Republic", "Congo", "DR Congo",
    "Uganda", "Kenya", "Tanzania", "Burundi", "Rwanda", "Somalia", "Djibouti", "Ethiopia", "Eritrea", 
    "Angola", "Mozambique", "Zambia", "Zimbabwe", "Malawi", "Namibia", "Botswana", "South Africa", "Lesotho", "Eswatini",
    "India", "Pakistan", "Bangladesh", "Philippines", "Vietnam", "Indonesia", "Turkey", 
    "Iran", "Iraq", "Afghanistan", "Syria", "Lebanon", "Jordan", "Saudi Arabia", "Yemen", 
    "Oman", "United Arab Emirates", "Qatar", "Bahrain", "Kuwait", 
    "Thailand", "Myanmar", "Cambodia", "Laos", "Malaysia", "Brunei"}

def count_unique_country_mentions(text, country_set):
    unique_countries_mentioned = set()
    text_lower = text.lower()  # Convert the text to lower case to handle case insensitivity
    for country in country_set:
        if country.lower() in text_lower:
            unique_countries_mentioned.add(country)
    return len(unique_countries_mentioned)

df_combined['eu_mentions'] = df_combined['body_resolved'].apply(lambda text: count_unique_country_mentions(text, eu_countries))
df_combined['noneu_mentions'] = df_combined['body_resolved'].apply(lambda text: count_unique_country_mentions(text, non_eu_countries))
df_combined['gn_mentions'] = df_combined['body_resolved'].apply(lambda text: count_unique_country_mentions(text, global_north_countries))
df_combined['gs_mentions'] = df_combined['body_resolved'].apply(lambda text: count_unique_country_mentions(text, global_south_countries))

all_svos_flat = [
    (svo, df_combined.loc[i, 'outlet'], df_combined.loc[i, 'eu_mentions'], df_combined.loc[i, 'noneu_mentions'],
     df_combined.loc[i, 'gn_mentions'], df_combined.loc[i, 'gs_mentions'], i, 1)
    for i, svos_list in enumerate(df_combined['svos']) for svo in svos_list
]

columns = ['SVO', 'Outlet', 'EU_Mentions', 'NonEU_Mentions', 'GN_Mentions', 'GS_Mentions', 'Article_Index', 'Count']
svo_df = pd.DataFrame(all_svos_flat, columns=columns)

aggregate_functions = {
    'EU_Mentions': 'sum',
    'NonEU_Mentions': 'sum',
    'GN_Mentions': 'sum',
    'GS_Mentions': 'sum',
    'Outlet': lambda x: ','.join(set(x)),
    'Article_Index': 'nunique',
    'Count': 'sum'  # Add this line to sum up the counts
}
svo_aggregated_df = svo_df.groupby('SVO').agg(aggregate_functions).reset_index()

# Adjusting the 'Outlet' column to replace multiple outlets with 'both' if there's more than one unique outlet
svo_aggregated_df['Outlet'] = svo_aggregated_df['Outlet'].apply(lambda x: 'both' if ',' in x else x)
# Rename 'Article_Index' to 'n_articles' for clarity
svo_aggregated_df.rename(columns={'Article_Index': 'n_articles', 'Count': 'Total_Count'}, inplace=True)

svo_aggregated_df.to_csv('final_df_coref_frag.csv', index=False)
