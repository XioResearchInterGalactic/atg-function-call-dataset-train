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

with open('project-1-at-2024-03-09-20-48-e5a49ce8.json') as f:
    dataset = json.load(f)
    print(len(dataset))

ls = LabelStudio(
    url=LABEL_STUDIO_URL,
    api_key=LABEL_STUDIO_API_KEY,
    single_turn_web_search_project_id=LABEL_STUDIO_SINGLE_TURN_WEB_SEARCH_PROJECT_ID,
    multi_turn_web_search_project_id=LABEL_STUDIO_MULTI_TURN_WEB_SEARCH_PROJECT_ID
)
bing = BingClient(BING_KEY)

#%%
def fetch_response(query):
    response = bing.query(query)
    results = [asdict(result) for result in response]
    return results[:5]

def parse_annotations(annotations) -> dict:
    annotations = json.loads(annotations)['tools']
    new_annotations = []
    for annotation in annotations:
        new_annotations.append({
            "tools": [
                {
                    "name": "get_web_search_result",
                    "arguments": {
                        "query": annotation['arguments']['query']
                    }
                }
            ]
        })
    return new_annotations

def generate_corpus_qa(conversation):

    function_response = next((item['content'] for item in conversation if item["role"] == "function_response"), None)
    try:
        information_list = json.loads(function_response)
    except Exception as e:
        print(function_response)
        return ""
    information = ['Information:\t' + inf['snippet'].strip() for inf in information_list]
    information = '\n'.join(information)
    question = next((item['content'] for item in conversation if item["role"] == "user"), None)

    with open('corpus-instruction.txt', "r") as template_file:
        template_content = template_file.read()
    template = Template(template_content)
    prompt = template.substitute(
        information=information,
        question=question
    )

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama2:13b",
            "prompt": prompt,
            'stream': False
        }
    )
    response = response.json()['response']
    first_index = response.find('{')
    last_index = response.rfind('}')
    response = response[first_index:last_index + 1]
    response = response.replace('\n', '\\n').replace("{\\n", "{").replace("\"\\n}", "\"}").replace("true,\\n\"", "true, \"")
    try:
        return json.loads(response)['answer']
    except Exception as e:
        print(response)
        return ""
    

def generate(conversation):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama2:13b",
            "prompt": f"User: {conversation[0]['content']}\nResponse: ",
            'stream': False
        }
    )
    response = response.json()['response']
    return response.strip()

#%%
for row in tqdm(dataset):
    annotations = row['annotations'][0]['result'][0]['value']['text'][0]
    row = row['data']    
    annotations = parse_annotations(annotations)
    conversation = []
    ui = {
        "user_1": row['user_1'],
        'function_call_1': '',
        'function_response_1': {},
        'function_call_2': '',
        'function_response_2': {},
        'assistant_1': '',
        'user_2': ''
    }
    conversation.append({'role': 'user', 'content': row['user_1']})
    i = 1
    for annotation in annotations:
        conversation.append({'role': 'function_call', 'content': json.dumps(annotation)})
        ui['function_call_' + str(i)] = json.dumps(annotation)
        search_result = fetch_response(annotation['tools'][0]['arguments']['query'])
        if search_result is not None:
            ui['function_response_' + str(i)] = {f'{ind}': search_result[ind]['snippet'] for ind in range(len(search_result))}
        conversation.append({'role': 'function_response', 'content': json.dumps(search_result)})
        i += 1
    if len(annotations) > 0:
        assistant_response = generate_corpus_qa(conversation)
        conversation.append({'role': 'assistant', 'content': assistant_response})
        ui['assistant_1'] = assistant_response
    else:
        assistant_response = generate(conversation)
        conversation.append({'role': 'assistant', 'content': assistant_response})
        ui['assistant_1'] = assistant_response
    
    if 'conversation' in row:
        last_user = next((item for item in reversed(row['conversation']) if item["role"] == "user"), None)
        conversation.append({'role': 'user', 'content': last_user['content']})
        ui['user_2'] = last_user['content']
        prediction = json.dumps({
            "tools": [
                {
                    "name": "get_web_search_result",
                    "arguments": {
                        "query": row['search_terms']['q2_search_term']
                    }
                }
            ]
        }, indent=2)
    else:
        prediction = ""
    
    row['conversation'] = conversation
    ui = [{'key': key, 'value': value} for key, value in ui.items()]
    row['ui'] = ui
    ls.log_multi_turn_web_search(row, assistant_response, prediction)

#%%
