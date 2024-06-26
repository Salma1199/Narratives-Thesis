# Creating dehumanization scorer

# import pandas as pd
# import transformers
# import torch
# from transformers import pipeline
# import ast
# from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


# final_df = pd.read_csv('final_clusters_data')
# results_df = pd.read_csv('approach1_final.csv')
# df = pd.read_csv('final_clustered_data')
# # print(len(df))

# # print(df['hdb_string_clusters'].nunique())
# # final_df = df


# # # Assuming final_df is your DataFrame containing the SVOs and clusters
# # filtered_df = final_df[final_df['hdb_string_clusters'] != -1]
# # cluster_texts = filtered_df.groupby('hdb_string_clusters')['string_SVO_col'].apply(' '.join).reset_index()
# # print(len(cluster_texts))

# # # Assuming you have a suitable model loaded, for example, a fine-tuned LLaMA model on evaluative tasks
# # model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# # tokenizer = AutoTokenizer.from_pretrained(model_id)
# # model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

# # # pipeline = transformers.pipeline(
# # #     "text-generation",
# # #     model=model_id,
# # #     model_kwargs={"torch_dtype": torch.bfloat16},
# # #     # device_map="auto",
# # # )
# # generator = pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, tokenizer = tokenizer)

# # from transformers import pipeline
# # import re

# # def process_cluster(svo_text):
# #     messages = [
# #         {"role": "system", "content": "You are linguistic expert, specifically in dehumanizing/othering language detection"},
# #         {"role": "user", "content": f"""
# #         I have analyzed texts describing migrants from UK news articles and need your help assessing if the language used is dehumanizing. Below is a cluster: '{svo_text}'. Please rate the dehumanizing nature of this cluster on a scale from 0 to 1, where 0 is not dehumanizing at all and 1 is extremely dehumanizing.
# #         Consider the following criteria for dehumanization:
# #         - **Extreme Dehumanization** (0.9-1): Language that likens migrants to animals or insects, explicitly diminishing the humanity of individuals (e.g., 'parasites', 'leeches').
# #         - **High Dehumanization** (0.75-0.89): Comparisons to natural disasters (e.g., 'flood', 'wave', 'tide', 'overflow', 'drown', 'pouring') or illegal connotations (e.g., 'illegal', 'unlawful').
# #         - **Moderate Dehumanization** (0.6-0.74): Treating migrants as inanimate objects (e.g., 'boats', 'docked', 'distribute', 'allocate', 'stranded', 'deport', 'smuggle').
# #         - **Lower Dehumanization** (0.4-0.59): Reducing migrants to mere numbers (e.g., 'thousands', 'millions', 'hundreds', 'tens', 'double', 'triple' ).
# #         Please provide a single decimal number that reflects your assessment for the language used in this cluster, starting your response with "I rate the dehumanizing nature of this language as X.XX", replacing X.XX with your rating between 0 and 1 ".
# #         """}
# #     ]
# # #     terminators = [
# # #     pipeline.tokenizer.eos_token_id,
# # #     pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
# # # ]

# #     outputs = generator(
# #         messages,
# #         max_new_tokens=256,
# #         #eos_token_id=terminators,
# #         do_sample=False,
# #         temperature=0.6,
# #         top_p=0.9,
# #     )
# #     return outputs[0]['generated_text'][-1]

# # cluster_texts['dehumanization_score'] = cluster_texts['string_SVO_col'].apply(process_cluster)

# # def extract_dehumanization_score(row):
# #     try:
# #         # Evaluate the string to a dictionary (if it's already a dictionary, this step is not needed)
# #         if isinstance(row, str):
# #             dict_content = ast.literal_eval(row)
# #         else:
# #             dict_content = row
        
# #         # Extract the 'content' from the dictionary
# #         content_text = dict_content['content']
        
# #         # Regular expression to find the floating number after the specific phrase
# #         pattern = r"the dehumanizing nature of this language as (\d+\.\d{2})"
# #         match = re.search(pattern, content_text)
# #         if match:
# #             return float(match.group(1))  # Converts the matched string to float
# #     except (SyntaxError, ValueError, KeyError):
# #         return None
# #     return None  

# # # Apply the function to the appropriate column. Ensure 'generated_text' is the correct column name
# # cluster_texts['predicted_score'] = cluster_texts['dehumanization_score'].apply(extract_dehumanization_score)

# # cluster_texts.to_csv('dehumanization_scores2')

# import pandas as pd
# import transformers
# import torch
# from transformers import pipeline
# import ast
# #from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# final_df = pd.read_csv('final_clusters_data')
# results_df = pd.read_csv('approach1_final.csv')
# df = pd.read_csv('final_clustered_data')
# print(df['hdb_string_clusters'].nunique())
# final_df = df

# # Assuming final_df is your DataFrame containing the SVOs and clusters
# filtered_df = final_df[final_df['hdb_string_clusters'] != -1]
# print(len(filtered_df))

# # Group by cluster and concatenate SVO strings
# cluster_texts = filtered_df.groupby('hdb_string_clusters')['string_SVO_col'].apply(' '.join).reset_index()

# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# # tokenizer = AutoTokenizer.from_pretrained(model_id)
# # model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
# generator = pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16})

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device_map="cuda",
# )

# def process_cluster(svo_text):
#     messages = [
#         {"role": "system", "content": "You are linguistic expert, specifically in dehumanizing/othering language detection"},
#         {"role": "user", "content": f"""
#         I have analyzed texts describing migrants from UK news articles and need your help assessing if the language used is dehumanizing. Below is a cluster: '{svo_text}'. Please rate the dehumanizing nature of this cluster on a scale from 0 to 1, where 0 is not dehumanizing at all and 1 is extremely dehumanizing.
#         Consider the following criteria for dehumanization:
#         - **Extreme Dehumanization** (0.9-1): Language that likens migrants to animals or insects, explicitly diminishing the humanity of individuals (e.g., 'parasites', 'leeches').
#         - **High Dehumanization** (0.75-0.89): Comparisons to natural disasters (e.g., 'flood', 'wave', 'tide', 'overflow', 'drown', 'pouring') or illegal connotations (e.g., 'illegal', 'unlawful').
#         - **Moderate Dehumanization** (0.6-0.74): Treating migrants as inanimate objects (e.g., 'boats', 'docked', 'distribute', 'allocate', 'stranded', 'deport', 'smuggle').
#         - **Lower Dehumanization** (0.4-0.59): Reducing migrants to mere numbers (e.g., 'thousands', 'millions', 'hundreds', 'tens', 'double', 'triple' ) or describing them using business terms (e.g., cost, expense, pay, price, burden, investment, allowance, collect).
#         Please provide a single decimal number that reflects your assessment for the language used in this cluster, starting your response with "I rate the dehumanizing nature of this language as X.XX", replacing X.XX with your rating between 0 and 1 ".
#         """}
#     ]

#     terminators = [
#     pipeline.tokenizer.eos_token_id,
#     pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
# ]

#     outputs = generator(
#         messages,
#         max_new_tokens=256,
#         eos_token_id=terminators,
#         do_sample=False,
#         temperature=0.6,
#         top_p=0.9,
#     )
#     return outputs[0]['generated_text'][-1]

# cluster_texts['dehumanization_score'] = cluster_texts['string_SVO_col'].apply(process_cluster)

import pandas as pd
import transformers
import torch
from transformers import pipeline
import re

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

final_df = pd.read_csv('final_clusters_data')
results_df = pd.read_csv('approach1_final.csv')
df = pd.read_csv('final_clustered_data')
print(len(df))

final_df = df
filtered_df = final_df[final_df['hdb_string_clusters'] != -1]
cluster_texts = filtered_df.groupby('hdb_string_clusters')['string_SVO_col'].apply(' '.join).reset_index()

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map = "auto"
)

def process_cluster(svo_text):
    messages = [
        {"role": "system", "content": "You are linguistic expert, specifically in dehumanizing/othering language detection"},
        {"role": "user", "content": f"""
        I have analyzed texts describing migrants from UK news articles and need your help assessing the language used. Below is a cluster: '{svo_text}'. Please rate the language of this cluster on a scale from -1 to 1, where -1 is extremely dehumanizing, 0 is neutral, and 1 is extremely humanizing. 
        Structure your response as: "I rate the dehumanizing nature of this language as X.XX", replacing X.XX with your predicted score
        Consider the following criteria for rating the language:
        - **Extreme Dehumanization** (-1 to -0.9): Language that likens migrants to animals or insects, explicitly diminishing the humanity of individuals (e.g., 'parasites', 'leeches').
        - **High Dehumanization** (-0.89 to -0.75): Comparisons to natural disasters (e.g., 'flood', 'wave', 'tide', 'overflow', 'drown', 'pouring') or illegal connotations (e.g., 'illegal', 'unlawful').
        - **Moderate Dehumanization** (-0.74 to -0.6): Treating migrants as inanimate objects (e.g., 'boats', 'docked', 'distribute', 'allocate', 'stranded', 'deport', 'smuggle') or using business terms to describe them (e.g. allowance, pay, cost, unskilled)
        - **Lower Dehumanization** (-0.59 to -0.4): Reducing migrants to mere numbers (e.g., 'thousands', 'millions', 'hundreds', 'tens', 'double', 'triple').
        - **Neutral** (-0.4 to 0.4): Language that neither humanizes nor dehumanizes significantly, focusing on neutral descriptions or factual reporting.
        - **Moderate Humanization** (0.41 to 0.6): Language showing general support to migrants, framing them and migration at large in a positive light (e.g., 'support', 'tolerance', 'resilient', 'hardworking', 'well-being', 'skilled', 'diverse').
        - **High Humanization** (0.61 to 0.8): Language that portrays migrants in a familial or community context, and focuses on their emotional experience with mentions of feelings (e.g., 'community', 'hurt', 'human', 'rights', 'suffer', 'hardship', 'sad', 'hopeful', 'grateful').
        - **Extreme Humanization** (0.81 to 1): Language that emphasizes deep personal connections (e.g., 'brother', 'sister', 'daughter', 'family', 'connection').
        Please provide a single decimal number that reflects your assessment for the language used in this cluster.
        """}
    ]
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=251,
        eos_token_id=terminators,
        do_sample=False,
        temperature=0.6,
        top_p=0.9,
    )
    return outputs[0]['generated_text'][-1]

cluster_texts['dehumanization_score'] = cluster_texts['string_SVO_col'].apply(process_cluster)

cluster_texts = pd.read_csv("dehumanization_scores3")
pd.set_option('display.max_colwidth', None)
# extracting the float
import ast  # Import the ast module to safely evaluate strings containing Python expressions

def extract_dehumanization_score(row):
    try:
        if isinstance(row, str):
            dict_content = ast.literal_eval(row)
        else:
            dict_content = row
        content_text = dict_content['content']
    
        pattern = r"the dehumanizing nature of this language as (-?\d+\.\d{2})"
        match = re.search(pattern, content_text)
        if match:
            return float(match.group(1))  # Converts the matched string to float
    except (SyntaxError, ValueError, KeyError):
        return None
    return None

# Apply the function to the appropriate column. Ensure 'generated_text' is the correct column name
cluster_texts['predicted_score'] = cluster_texts['dehumanization_score'].apply(extract_dehumanization_score)

cluster_texts.to_csv('dehumanization_scores4')