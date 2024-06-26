# # importing relevant packagers 
# import pandas as pd
# import spacy
# # from fastcoref import FCoref
# # from fastcoref import spacy_component
# import coreferee, spacy
# nlp = spacy.load('en_core_web_trf')
# import spacy_transformers
# from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor, as_completed


# # reading saved csvs
# df_combined = pd.read_csv('df_combined_perigon')
# #svo_df = pd.read_csv('svo_df.csv')
# #sampled_df = pd.read_csv('sampled_df.csv')
# #filtered_df = pd.read_csv('filtered_df.csv')

# # Defining necessary processes and functions
# nlp = spacy.load('en_core_web_trf')
# nlp.add_pipe('coreferee')
# # https://spacy.io/universe/project/coreferee

# def resolve_coreferences(text):
#     doc = nlp(text)
#     resolved_tokens = [token.text for token in doc]

#     if doc._.coref_chains:
#         for chain in doc._.coref_chains:
#             main_mention = chain.mentions[0]
#             main_mention_text = ' '.join([doc[i].text for i in main_mention.token_indexes])
#             for mention in chain.mentions:
#                 if mention != main_mention:
#                     mention_indexes = mention.token_indexes
#                     mention_start = mention_indexes[0]
#                     mention_end = mention_indexes[-1]
#                     resolved_tokens[mention_start] = main_mention_text
#                     for i in range(mention_start + 1, mention_end + 1):
#                         resolved_tokens[i] = ''

#     resolved_text = ' '.join([token for token in resolved_tokens if token != ''])
#     return resolved_text

# # Function to apply coreference resolution in parallel
# def parallel_apply(data, func, workers=4):
#     results = {}
#     with ThreadPoolExecutor(max_workers=workers) as executor:
#         future_to_index = {executor.submit(func, text): i for i, text in data.items()}
#         for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Processing"):
#             i = future_to_index[future]
#             results[i] = future.result()
#     return pd.Series(results)

# df_combined['body_resolved'] = parallel_apply(df_combined['body'], resolve_coreferences)

# # Saving to a new df for easy access
# df_combined.to_csv('df_combined_coref.csv', index=False)


import pandas as pd
import spacy
import coreferee
import spacy_transformers
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load the large transformer model
nlp = spacy.load('en_core_web_trf')
nlp.add_pipe('coreferee')

# Function to resolve coreferences in a text
def resolve_coreferences(text):
    doc = nlp(text)
    resolved_tokens = [token.text for token in doc]

    if doc._.coref_chains:
        for chain in doc._.coref_chains:
            main_mention = chain.mentions[0]
            main_mention_text = ' '.join([doc[i].text for i in main_mention.token_indexes])
            for mention in chain.mentions:
                if mention != main_mention:
                    mention_indexes = mention.token_indexes
                    mention_start = mention_indexes[0]
                    mention_end = mention_indexes[-1]
                    resolved_tokens[mention_start] = main_mention_text
                    for i in range(mention_start + 1, mention_end + 1):
                        resolved_tokens[i] = ''

    resolved_text = ' '.join([token for token in resolved_tokens if token != ''])
    return resolved_text

# Function to apply coreference resolution in parallel
def parallel_apply(data, func, workers=4):
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {executor.submit(func, text): i for i, text in data.items()}
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Processing"):
            i = future_to_index[future]
            results[i] = future.result()
    return pd.Series(results)

# Read the CSV file
df_combined = pd.read_csv('df_combined_perigon_full')

# Drop rows where 'body' is NaN or not a string
df_combined = df_combined.dropna(subset=['body'])
df_combined = df_combined[df_combined['body'].apply(lambda x: isinstance(x, str))]

# Apply coreference resolution
df_combined['body_resolved'] = parallel_apply(df_combined['body'], resolve_coreferences)

# Save the results to a new CSV file
df_combined.to_csv('df_combined_coref_new.csv', index=False)
