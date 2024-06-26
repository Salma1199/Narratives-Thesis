import pandas as pd
import transformers
import torch
from transformers import pipeline
import ast
import re

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

final_df = pd.read_csv('final_clusters_data')
results_df = pd.read_csv('approach1_final.csv')
df = pd.read_csv('final_clustered_data')
print(df['hdb_string_clusters'].nunique())
final_df = df

# Assuming final_df is your DataFrame containing the SVOs and clusters
filtered_df = final_df[final_df['hdb_string_clusters'] != -1]
print(len(filtered_df))

# Group by cluster and concatenate SVO strings
cluster_texts = filtered_df.groupby('hdb_string_clusters')['string_SVO_col'].apply(' '.join).reset_index()

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map = "auto"
)

def process_cluster(svo_text):
    messages = [
        {"role": "system", "content": "You are an expert on migration studies"},
        {"role": "user", "content": f"""
        I have analyzed texts describing migrants from UK news articles and need your help and need your help to assess which higher-level topic each cluster falls under. Below are statements extracted from various texts: {svo_text}. For each cluster of extracts (row), your goal is to assign a label from the seven below that best captures the hgiher-level topic of those extracts. For each label, I have provided definitions and keywords with synonyms to aid your decision:
        1. Economics: Relating to financial situations and macroeconomic implications. Keywords: economic, labor, low-skilled, poor, skilled, unemployed, worker, job, take, cost, opportunity, employment, work, wage, food, pay, productivity
        2. Flows: Relating to volume of migration inflows and outflows, migration patterns, and border dynamics. Keywords: cross, channel, border, boom, flood, stock, mass, excessive, flight, airport, rescue, enter, route, number, hundred, million
        3. Crime and law: Relating to criminal activities and legal proceedings. Keywords: police, kill, crime, court
        4. Systems and policy: Relating to actions, individuals, or procesuring occurring in policy, legislative bodies, or government. Keywords: advisor, application, bill, department, minister, official, procedure, regulation, scheme, system, convention, order, demorat, report, republican, president, party, election, statement
        5. Integration: Relating to migrants' integration into host communities and lived experiences. Keywords: endure, remember, spend, consequence, house, shelter, leave, need help, nothing, anybody, experience, family, daughter
        6. Public attitudes: Relating to public perception to migrants, including characteristics used to differentiate treatment such as age, sex, ethnicity, nationality, religion, and culture. Keywords: culture, muslim, country origins (e.g. Afghan, Syrian), education, language, concern, problem, willingness)
        7. International issues/comparisons: Relating to issues in other countries that may be impacting migration in the UK, or mentioning other countries as benchmarks. Keywords: mentions of external conflicts and countries (e.g. Australia, Russia, Israel, Palestine)
        Select the number that best matches the main topic of each custer. Structure your response as follows "The predicted label is X", using exactly these words and replacing X with the label number you have assigned from 1 to 7 for each cluster. A label can be repeated for multiple clusters or not used at all, but each cluster (row) needs to be assigned to exactly one label.
        """}
    ]

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
    messages,
    max_new_tokens=521,
    eos_token_id=terminators,
    do_sample=False,
    temperature=0.6,
    top_p=0.9,
    )
    return outputs[0]['generated_text'][-1]

cluster_texts['topic_labels'] = cluster_texts['string_SVO_col'].apply(process_cluster)

import ast
import re

def extract_label(row):
    try:
        # Evaluate the string to a dictionary (if it's already a dictionary, this step is not needed)
        if isinstance(row, str):
            dict_content = ast.literal_eval(row)
        else:
            dict_content = row
        
        # Extract the 'content' from the dictionary
        content_text = dict_content['content']
        
        # Regular expression to find the integer after the specific phrase "The predicted label is"
        pattern = r"The predicted label is (\d+)"
        match = re.search(pattern, content_text)
        if match:
            return int(match.group(1))  # Converts the matched string to an integer
    except (SyntaxError, ValueError, KeyError):
        # Handles cases where the row is not a proper dictionary string, or key 'content' is missing
        return None
    return None  # Returns None if no match is found

cluster_texts['predicted_topic'] = cluster_texts['topic_labels'].apply(extract_label)
cluster_texts.to_csv('predicted_topic_labels2')
