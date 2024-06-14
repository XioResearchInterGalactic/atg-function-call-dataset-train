# %%
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
import wandb, os
import transformers
from datetime import datetime
import json
from datasets import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from string import Template
from datasets import load_dataset
import pandas as pd

# %%
dataset_id = 'lilacai/glaive-function-calling-v2-sharegpt'
functions = json.load(open('prompts/functions-v2.json', 'r'))

#%%
role_map = {
    'system': 'function_metadata',
    'human': 'user',
    'gpt': 'assistant',
    'tool': 'function_response',
}

#%%
def format_function_calling_dataset(idx, row):
    messages = []
    functions = []
    for message in row['conversations']:
        role = role_map[message['from']]
        message = {'role': role, 'content': message['value'].replace('<|endoftext|>', '').strip()}
        if message['content'].startswith('<functioncall>'):
            message['role'] = 'function_call'
            try:
                content = message['content'].replace('<functioncall>', '').strip()
                content = content.replace(', "arguments": \'', ', "arguments": ').replace("}'}", "}}").replace("\\'", "'").replace("]'}", "]}")
                message['content'] = json.dumps(json.loads(content), indent=2)
            except Exception as e:
                print(message['content'])
                return None
        if message['role'] == 'function_metadata':
            # get first index of "{"
            first_index = message['content'].find('{')
            last_index = message['content'].rfind('}')
            message['content'] = message['content'][first_index:last_index+1]
            message['content'] = message['content'].split('\n}')
            for function in message['content']:
                if len(function) < 10:
                    continue
                function = function + '}'
                function = json.loads(function)
                arguments = {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find information"
                        }
                    },
                    "required": []
                }
                if 'parameters' in function and function['parameters'] and len(list(function['parameters'].keys())) > 0 and 'properties' in function['parameters'] and function['parameters']['properties']:
                    arguments['properties'] = function['parameters']['properties']
                    if 'parameters' in function and function['parameters'] and len(list(function['parameters'].keys())) > 0 and 'required' in function['parameters'] and function['parameters']['required']:
                        arguments['required'] = function['parameters']['required']
                else:
                    arguments = None
                
                
                updated_functions = {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "value": function['name'],
                            "description": function['description']
                        },
                        "arguments": arguments
                    },
                    "required": ['name', 'arguments']
                }
                functions.append(updated_functions)
            
        if role != 'function_metadata':
            messages.append(message)
    
    return {
        'source_id': 'lilacai/glaive-function-calling-v2-sharegpt:' + str(idx),
        'functions': functions,
        'conversation': messages
    }

#%%
rows = []

#%%
dataset = []
for row in pd.read_csv('datasets/dataset.csv').to_dict(orient="records"):
    row['conversation'] = json.loads(row['conversation'])
    dataset.append(row)

# %%
manual_idx = 1
for row in tqdm(dataset):
    source_id = row['source_id']
    if source_id is None or source_id != source_id:
        source_id = 'manual:' + str(manual_idx)
        manual_idx += 1
    elif 'manual' in source_id:
        source_id = 'manual:' + str(manual_idx)
        manual_idx += 1
    if ':' not in source_id:
        idx = source_id.split('_')[-1]
        source_id = '_'.join(source_id.split('_')[:-1]) + ':' + idx
    rows.append({
        'source_id': source_id,
        'functions': functions,
        'conversation': row['conversation']
    })

#%%
dataset = load_dataset(dataset_id, split='train')
error_count = 0
for idx, row in tqdm(enumerate(dataset), total=len(dataset)):
    result = format_function_calling_dataset(idx, row)
    if result is not None:
        rows.append(result)
    else:
        error_count += 1

# %%
# save rows to jsonl
with open('datasets/dataset-v2.jsonl', 'w') as f:
    for row in tqdm(rows):
        f.write(json.dumps(row) + '\n')

#%%
error_count
# %%
