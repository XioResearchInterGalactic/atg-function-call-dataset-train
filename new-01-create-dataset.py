# %%
import os
import json
from tqdm import tqdm
from datasets import load_dataset
from dotenv import load_dotenv
import requests
import random
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

# %% LOAD ENV VARIABLES
load_dotenv()
dataset_id = 'lilacai/glaive-function-calling-v2-sharegpt'
directus_url = os.getenv('DIRECTUS_URL')
directus_email = os.getenv('DIRECTUS_EMAIL')
directus_password = os.getenv('DIRECTUS_PASSWORD')
functions_json = json.load(open('prompts/functions-v2.json', 'r'))
target_dataset_repo = 'MerlynMind/ATG_Function_Call_SFT_V1'
filter_directus_rows = False

#%% GLAIVE DATASET ROLE MAP
glaive_role_map = {
    'system': 'function_metadata',
    'human': 'user',
    'gpt': 'assistant',
    'tool': 'function_response',
}

#%% DIRECTUS API
def get_directus_access_token(url: str, email: str, password: str) -> str:
    url = url + "/auth/login"
    body = {"email": email, "password": password}
    response = requests.post(url, json=body)
    return response.json()["data"]["access_token"]

def get_directus_conversations(url: str, access_token: str) -> list:
    url = url + "/items/conversations?limit=5000"
    if filter_directus_rows:
        url += "&filter[user_id][_eq]=5f7aa8b8-fb98-4953-87fa-e7c443aeb9af"
    headers = {"Authorization": "Bearer " + access_token}
    return requests.get(url, headers=headers).json()['data']

#%% FORMAT FUNCTION CALLING DATASET
def format_function_calling_dataset(idx, row):
    messages = []
    functions = []
    for message in row['conversations']:
        role = glaive_role_map[message['from']]
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
        'source_id': dataset_id + ':' + str(idx),
        'category': dataset_id,
        'functions': random.sample(functions, len(functions)),
        'conversation': messages
    }

#%% LOGIN TO DIRECTUS
access_token = get_directus_access_token(directus_url, directus_email, directus_password)

#%% GET DIRECTUS CONVERSATIONS
directus_dataset = get_directus_conversations(directus_url, access_token)
print('Directus dataset length:', len(directus_dataset))

#%% FORMAT DIRECTUS DATASET
dataset = list(map(lambda x: {
    'source_id': x['source_id'],
    'category': x['category'],
    'functions': random.sample(functions_json, len(functions_json)),
    'conversation': x['conversation']
}, directus_dataset))

#%% FORMAT SOURCE ID
rows = []
manual_idx = 1
for row in tqdm(dataset):
    source_id = row['source_id']
    if source_id is None or source_id.strip() == '' or source_id != source_id or 'manual' in source_id:
        source_id = 'manual:' + str(manual_idx)
        manual_idx += 1
    if ':' not in source_id:
        idx = source_id.split('_')[-1]
        source_id = '_'.join(source_id.split('_')[:-1]) + ':' + idx
    row['source_id'] = source_id
    rows.append(row)
dataset = rows

#%% FORMAT FUNCTION CALLING DATASET
fdataset = load_dataset(dataset_id, split='train')
error_count = 0
for idx, row in tqdm(enumerate(fdataset), total=len(fdataset)):
    result = format_function_calling_dataset(idx, row)
    if result is not None:
        dataset.append(result)
    else:
        error_count += 1
print(f"Error count: {error_count}")

#%% CLEAN UP
dataset2 = []
skip_outer = False
skip_inner = False
for row in tqdm(dataset):
    if row['functions'] is None or len(row['functions']) == 0:
        continue
    new_conv = []
    for message in row['conversation']:
        if message['role'] == 'function_call':
            content_json = json.loads(message['content'])

            # check if function_call has arguments as list
            if 'arguments' in content_json and isinstance(content_json['arguments'], list):
                skip_outer = True 
                break

            # check if function_call has arguments as None
            if 'arguments' in content_json and content_json['arguments'] is None:
                new_message = {
                    'role': 'function_call',
                    'content': json.dumps(content_json, indent=2)
                }
                new_conv.append(new_message)
                continue

            # check if function_call has arguments as dictionary
            if 'arguments' in content_json and content_json['arguments'] is not None:
                keys = list(content_json['arguments'].keys())
                arguments = {}
                for key in keys:
                    value = content_json['arguments'][key]
                    if not isinstance(value, str):
                        skip_inner = True
                        break
                    arguments[key] = value.strip()
                if skip_inner:
                    print('Skipping due to error')
                    skip_inner = False
                    skip_outer = True
                    break
                content_json['arguments'] = arguments
                new_message = {
                    'role': 'function_call',
                    'content': json.dumps(content_json, indent=2)
                }
                new_conv.append(new_message)
            else:
                new_conv.append(message)
        else:
            new_conv.append(message)
    
    if skip_outer:
        print('Skipping due to error')
        skip_outer = False
        continue
    
    # find idx of function_call messages
    function_call_idx = []
    for idx, message in enumerate(new_conv):
        if message['role'] == 'function_call':
            function_call_idx.append(idx)

    # check if function_call is after user message or function_response
    for idx in function_call_idx:
        if idx == 0:
            skip_outer = True
            break # function_call cannot be first message
        if new_conv[idx-1]['role'] != 'user' and new_conv[idx-1]['role'] != 'function_response':
            skip_outer = True
            break

    # find idx of function_response messages
    function_response_idx = []
    for idx, message in enumerate(new_conv):
        if message['role'] == 'function_response':
            function_response_idx.append(idx)

    # check if function_response is after function_call
    for idx in function_response_idx:
        if idx == 0:
            skip_outer = True
            break # function_response cannot be first message
        if new_conv[idx-1]['role'] != 'function_call':
            skip_outer = True
            break
    
    if skip_outer:
        print('Skipping due to error')
        skip_outer = False
        continue

    # get last index where role = assistant
    assistant_idx = -1
    for idx, message in enumerate(new_conv):
        if message['role'] == 'assistant':
            assistant_idx = idx

    if assistant_idx == -1:
        continue

    # keep messages until last assistant message
    new_conv = new_conv[:assistant_idx+1]

    row['conversation'] = json.dumps(new_conv)
    dataset2.append(row)

dataset = dataset2

#%% AUGMENT SOME NEW CONVERSATIONS
only_assistant_responses = []
function_call_responses = []
annotated_dataset = list(filter(lambda x: 'chat_alpaca' in x['source_id'] or 'manual' in x['source_id'], dataset))
for row in tqdm(annotated_dataset):
    conversation = json.loads(row['conversation'])
    if conversation[1]['role'] == 'assistant':
        only_assistant_responses.append(conversation[0:2])
    else:
        assistant_idx = -1
        for idx, message in enumerate(conversation):
            if message['role'] == 'assistant':
                assistant_idx = idx
                break
        function_call_responses.append(conversation[0:assistant_idx+1])

new_conversations = []
for _ in tqdm(range(100)):
    num_elements = random.randint(2, 5)
    assistant_messages = random.sample(only_assistant_responses, num_elements)
    conv = []
    for message in assistant_messages:
        conv.extend(message)
    function_call_message = random.choice(function_call_responses)
    conv.extend(function_call_message)
    new_conversations.append(conv)

i = 1
for conv in new_conversations:
    dataset.append({
        'source_id': f"augment:{i}",
        'category': 'Mixed',
        'functions': functions_json,
        'conversation': json.dumps(conv)
    })
    i += 1

#%% PRINT UNIQUE CATEGORIES
unique_categories = set()
for row in dataset:
    unique_categories.add(row['category'])
print(unique_categories)

#%% COUNT ANNOTATED
annotated_count = 0
for row in dataset:
    if row['category'] != 'lilacai/glaive-function-calling-v2-sharegpt':
        annotated_count += 1
print('Annotated count:', annotated_count)

#%% BALANCE THE DATASET TO INCREASE WEIGHT OF ANNOTATED DATASET
dataset2 = []
count = 0
for row in dataset:
    if row['category'] == 'lilacai/glaive-function-calling-v2-sharegpt' and count <= annotated_count * 2:
        count += 1
        dataset2.append(row)
    elif row['category'] != 'lilacai/glaive-function-calling-v2-sharegpt':
        dataset2.append(row)
print(len(dataset2))
dataset = dataset2

#%% CONVERT CONVERSATION TO OBJECTS
dataset2 = []
for row in tqdm(dataset):
    row['conversation'] = json.loads(row['conversation'])
    row['functions'] = json.dumps(row['functions'])
    dataset2.append(row)
dataset = dataset2

#%% GROUP DATASET BY CATEGORY
category_dataset = {}
for row in dataset:
    category = row['category']
    if category not in category_dataset:
        category_dataset[category] = []
    category_dataset[category].append(row)

#%%
for category in category_dataset:
    print('Category:', category, 'Count:', len(category_dataset[category]))

#%% SPLIT TRAIN TEST SET. TAKE 5% FROM lilacai CATEGORY AND TAKE 10% FROM THE REST OF THE CATEGORY
train_dataset = []
test_dataset = []
for category in category_dataset:
    rows = category_dataset[category]
    if category == 'lilacai/glaive-function-calling-v2-sharegpt':
        train, test = train_test_split(rows, test_size=0.01, random_state=42)
    else:
        train, test = train_test_split(rows, test_size=0.10, random_state=42)
    print('Category:', category, 'Train:', len(train), 'Test:', len(test))
    train_dataset.extend(train)
    test_dataset.extend(test)
print('Train:', len(train_dataset), 'Test:', len(test_dataset))

#%% SHUFFLE ROWS
train_dataset = random.sample(train_dataset, len(train_dataset))
test_dataset = random.sample(test_dataset, len(test_dataset))

#%% SAVE DATASET TO FILE
with open('datasets/train-dataset.json', 'w') as f:
    json.dump(train_dataset, f, indent=2)
with open('datasets/validation-dataset.json', 'w') as f:
    json.dump(test_dataset, f, indent=2)

#%% UPLOAD TO HUGGINGFACE DATASET
train_dataset = Dataset.from_list(train_dataset)
test_dataset = Dataset.from_list(test_dataset)
dataset_dict = DatasetDict({"train": train_dataset, "validation": test_dataset})
dataset_dict.push_to_hub(target_dataset_repo, token=os.getenv('HF_TOKEN'))
