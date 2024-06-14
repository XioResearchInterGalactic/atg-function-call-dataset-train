#%%
from dotenv import load_dotenv
import os
from labelstudio import LabelStudio
from tqdm import tqdm
import json
from bing_client import BingClient
from dataclasses import asdict
import requests
from string import Template

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY")
LABEL_STUDIO_SINGLE_TURN_WEB_SEARCH_PROJECT_ID = os.getenv("LABEL_STUDIO_WEB_SEARCH_SINGLE_TURN_PROJECT_ID")
LABEL_STUDIO_MULTI_TURN_WEB_SEARCH_PROJECT_ID = os.getenv("LABEL_STUDIO_WEB_SEARCH_MULTI_TURN_PROJECT_ID")
BING_KEY = os.getenv("BING_KEY")

#%%
with open('project-2-at-2024-03-11-00-24-9ee67a1c.json') as f:
    dataset = json.load(f)
    print(len(dataset))

#%%
ds = dataset[54:55]
ds = dataset[0:1]
new_ds = []
id = 1
for row in tqdm(dataset):
    conversation = row['data']['conversation']
    assistant_index = next((i for i, item in enumerate(conversation) if item["role"] == "assistant"), None)
    annotations = row['annotations'][0]['result']
    assistant_1 = next((item for i, item in enumerate(annotations) if item["from_name"] == "assistant_1"), None)
    assistant_1 = assistant_1['value']['text'][0]
    conversation[assistant_index] = {
        "role": "assistant",
        "content": assistant_1
    }
    function_call = next((item for i, item in enumerate(annotations) if item["from_name"] == "function_call_3"), None)
    function_call = function_call['value']['text'][0]
    if len(function_call) > 0:
        conversation.append({
            "role": "function_call",
            "content": function_call
        })
    print(json.dumps(row, indent=2))
    source_id = f"manual:{row['id']}"
    if row['data']['category'] != "manual":
        source_id = f"MerlynMind/RAG_Current_Events_v1_20240220:{row['data']['row_id']}"
    date = None
    if 'date' in row['data']:
        date = row['data']['date']
    
    # loop entries in conversation including index
    for i, item in enumerate(conversation):
        if item["role"] == "function_call" or item["role"] == "function_response":
            function_call = item['content']
            if len(function_call) > 0:
                function_call = json.loads(function_call)
            conversation[i] = {'role': item['role'], 'content': json.dumps(function_call, indent=2)}

    new_ds.append({
        'id': id,
        'source_id': source_id,
        'date': date,
        'category': row['data']['category'],
        'conversation': conversation,
    })
    id += 1

#%%
new_ds = json.dumps(new_ds, indent=2)
with open('new_ds.json', 'w') as f:
    f.write(new_ds)

# %%
