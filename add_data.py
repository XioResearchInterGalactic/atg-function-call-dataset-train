#%%
from bing_client import BingClient
from dotenv import load_dotenv
import os
import json

load_dotenv()
BING_KEY = os.getenv("BING_KEY")
bing = BingClient(BING_KEY)

#%%
dataset = json.load(open("dataset/dataset.json", "r"))

#%%
def add_single_turn(question, search_queries):
    id = sorted([d["id"] for d in dataset])[-1]
    manual = [d for d in dataset if "manual" in d["source_id"]]
    source_id = int(sorted([d["source_id"] for d in manual])[-1].split(":")[1]) + 1
    conversation = [
        {
            "role": "user",
            "content": question
        }
    ]
    tool_arg_map = {
        "get_web_search_result": 'query',
        'get_python_math_result': 'expression'
    }
    for search_query in search_queries:
        conversation.append({
            "role": "function_call",
            "content": "{\n  \"tools\": [\n    {\n      \"name\": \"" + search_queries['tool'] +  "\",\n      \"arguments\": {\n        \"" + tool_arg_map[search_query['tool']] + "\": \"" + search_query['query'] + "\"\n      }\n    }\n  ]\n}"
        })
        search_result = None
        if search_query['tool'] == "bing":
            search_result = json.dumps(bing.search(search_query['query']), indent=2)
        elif search_query['tool'] == "python_math":
            try:
                exec(search_query['query'])
                search_result = json.dumps(result, indent=2)
            except Exception as e:
                print('failure on ' + question)
                return
        conversation.append({
            "role": "function_response",
            "content": search_result
        })
    
    dataset.append({
        "id": id,
        "source_id": "manual:343",
        "date": None,
        "category": f"manual:{source_id}",
        "conversation": conversation,
    })

#%%
    add_single_turn("What is the capital of France?", [