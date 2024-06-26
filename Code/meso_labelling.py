import pandas as pd
import transformers
import torch
from transformers import pipeline
import ast
#from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device_map = "auto"
# )

# def process_cluster(svo_text):
#     messages = [
#         {"role": "system", "content": "You are an expert on migration studies. I have extracted subject-verb-object (SVO) triples from news articles, and have clustered those SVOs into 43 clusters."},
#         {"role": "user", "content": f"""
#         I want to attach a label to the set of SVOs attached, by selecting the label that most appropriately captures the high-level idea of the SVOs provided. Consider the following statements extracted from various texts: {svo_text}. Which one of the following narratives best describes the main narrative of these statements?
#         1. Migrants cheat systems
#         2. Migrants are linked to crime/antisocial/problematic behavior
#         3. The UK needs to deter migrants
#         4. Systems fail migrants
#         5. Policy failure is causing problems in the UK migration system
#         6. Better policy/enforcement is needed to solve a migration problem
#         7. Migration policies are flawed or will not work
#         8. Levels of immigration/net migration to the UK are too high
#         9. Specific types of migration are too high
#         10. Migrants may soon arrive in the UK as a result of issues elsewhere
#         11. Migrants travel by irregular means
#         12. Migration (/specific type of migration) is out of control
#         13. Large numbers of people from a specific location are migrating to the UK
#         14. Levels of immigration/net migration to (a) non-UK country(s) are too high (general)
#         15. Migrants are a drain on public finances
#         16. Migrants negatively affect local finances
#         17. The economy/specific sectors need(s) migrants
#         18. Specific sectors/roles are dominated by migrants
#         19. Migrants have a positive impact on public finances
#         20. Migrants receive priority treatment over British citizens
#         21. Migrants are not well integrated
#         22. The UK provides support/welfare/accommodation for migrants
#         23. The distribution of migrants around the UK is a problem
#         24. Migrants (or a specific migrant) are strong/a powerful force
#         25. Migrants are well integrated
#         26. Migrants (or a specific migrant) are mistreated
#         27. Migrants (or a specific migrant) are vulnerable
#         28. The public is opposed to migration
#         29. The public is concerned about a specific migrant-related issue or group
#         30. The public is less opposed to/not opposed to/supportive of migration
#         31. Migration, or responses to it create tensions or undermine social cohesion
#         32. The public supports a particular policy/plan related to migration
#         33. The public opposes a particular policy/plan related to migration
#         34. Migration is significantly affecting another country(/s)
#         35. The UK compares poorly to another country on migration issues
#         36. He UK compares well to another country on migration issues
#         37. There are lessons to be learned from another country’s management if migration
#         Structure your response as follows "The predicted label is X", usiong exactly these words and replacing X with the label number you have assigned from 1 to 37 for each cluster. A label can be repeated for multiple clusters or not used at all.
#         """}
#     ]

#     terminators = [
#         pipeline.tokenizer.eos_token_id,
#         pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
#     ]

#     outputs = pipeline(
#     messages,
#     max_new_tokens=521,
#     eos_token_id=terminators,
#     do_sample=False,
#     temperature=0.6,
#     top_p=0.9,
#     )
#     return outputs[0]['generated_text'][-1]

# cluster_texts['meso_labels'] = cluster_texts['string_SVO_col'].apply(process_cluster)

import ast
import re

cluster_texts = pd.read_csv('meso_labels2') 

def extract_label(row):
    try:
        # Convert row to dictionary if it's a string
        if isinstance(row, str):
            dict_content = ast.literal_eval(row)
        else:
            dict_content = row
        
        content_text = dict_content['content']
        pattern = r"I predict that the label is (\d+)"
        match = re.search(pattern, content_text)
        if match:
            label_number = int(match.group(1))
            # Check if the label number is within the valid range (1-37)
            if 1 <= label_number <= 37:
                return label_number
    except (SyntaxError, ValueError, KeyError):
        return None
    return None 

cluster_texts['predicted_meso'] = cluster_texts['meso_labels'].apply(extract_label)
cluster_texts.to_csv('meso_labels2')