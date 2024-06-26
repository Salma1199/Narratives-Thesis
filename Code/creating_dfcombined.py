# Creating df_combined using PERIGON articles (i.e. without "extracted_data key")

import os
import json
import pandas as pd

# DATAFRAME FOR TELEGRAPH
# Define an empty list to store the extracted data
articles_data = []

# Path to the folder containing JSON files
folder_path = "articles"

# Set to keep track of processed article bodies
processed_bodies = set()

# Iterate through each file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".json"):
        file_path = os.path.join(folder_path, file_name)
        print(f"Processing file: {file_path}")  # Debugging statement
        
        # Load JSON data from the file
        try:
            with open(file_path, "r") as f:
                json_data = json.load(f)
            print(f"Loaded JSON data: {json_data}")  # Debugging statement
        except json.JSONDecodeError:
            print(f"Skipping file {file_name} as it does not contain valid JSON data.")
            continue
        
        # Extract required information directly from the JSON data
        body = json_data.get("text", "")
        
        # Check if the article body has already been processed
        if body in processed_bodies:
            print(f"Skipping duplicate article with body: {body[:30]}...")  # Print first 30 characters for reference
            continue

        # Append extracted data to the list
        articles_data.append({
            "author": ", ".join(json_data.get("authors", [])),
            "title": json_data.get("title", ""),
            "date": json_data.get("publish_date", ""),
            "body": body
            # Add additional tags as needed
        })
        
        # Add the body to the set of processed bodies
        processed_bodies.add(body)

# Convert the list to a DataFrame
df= pd.DataFrame(articles_data)
print(len(df))
      
# Convert "date" column to datetime objects with specified format and coerce parsing errors
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

# Understanding time range of data

minimum_date = df['date'].min()
maximum_date = df['date'].max()

print("Minimum Date:", minimum_date)
print("Maximum Date:", maximum_date)

df_Telegraph = df[['date', 'title', 'body']]

import re

def clean_text(text):
    # Remove newline characters
    cleaned_text = text.replace('\n', ' ')
    
    # Remove extra whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    # Remove other special characters but keep periods, commas, and question marks
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s.,?]', '', cleaned_text)
    
    return cleaned_text
df_Telegraph['body'] = df['body'].apply(clean_text)
df_Telegraph = df_Telegraph.sample(frac=1, random_state=42)

# DATAFRAME FOR GUARDIAN DATA

file_path = "guardian_articles.csv"
df_guardian = pd.read_csv(file_path)

# Understanding time range of data
minimum_date = df_guardian['Date'].min()
maximum_date = df_guardian['Date'].max()

print("Minimum Date:", minimum_date)
print("Maximum Date:", maximum_date)
df_guardian.rename(columns={'Date': 'date', 'Title': 'title', 'Body': 'body'}, inplace=True)

df_guardian.dropna(subset=['body'], inplace=True)
df_guardian['body'] = df_guardian['body'].apply(clean_text)
# Randomly shuffle the rows of the DataFrame
df_guardian = df_guardian.sample(frac=1, random_state=42)
df_guardian_sampled = df_guardian.sample(n=6027, random_state=42)
print(len(df_guardian_sampled))

df_guardian_sampled['outlet'] = 'guardian'
df_Telegraph['outlet'] = 'telegraph'
df_guardian_sampled['date'] = pd.to_datetime(df_guardian_sampled['date'])
df_Telegraph['date'] = pd.to_datetime(df_Telegraph['date'])


# Select only the columns of interest from each DataFrame
df_guardian_selected = df_guardian_sampled[['date', 'title', 'body', 'outlet']]
df_telegraph_selected = df_Telegraph[['date', 'title', 'body', 'outlet']]

# Saving dataframes
df_telegraph_selected.to_csv('df_telegraph_perigon')
df_guardian_selected.to_csv('df_guardian_perigon')

# Vertically concatenateataFrames
df_combined = pd.concat([df_guardian_selected, df_telegraph_selected], ignore_index=True)
print(len(df_combined))

df_combined.to_csv('df_combined_perigon')