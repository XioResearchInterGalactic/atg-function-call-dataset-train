#%%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bing_client import BingClient
from dotenv import load_dotenv
import os
import json
from dataclasses import asdict
import traceback

load_dotenv()
BING_KEY = os.getenv("BING_KEY")
bing = BingClient(BING_KEY)

#%%
ft_model_id = '/home/raquib/function-dataset/outputs/output_models_lr_0.000005/checkpoint-666'
tokenizer = AutoTokenizer.from_pretrained(ft_model_id, add_bos_token=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    ft_model_id,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
functions = open("prompts/functions.txt", "r").read()
instruction = open("prompts/instruction.txt", "r").read()
chat_template = open("prompts/template.txt", "r").read()
tokenizer.chat_template = chat_template


#%%
def generate(messages: list):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    model_inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    print('=====PROMPT=====')
    print(prompt)
    print('-----RESPONSE-----')
    with torch.no_grad():
        result = model.generate(**model_inputs, max_new_tokens=250)[0]
        result = tokenizer.decode(result, skip_special_tokens=True)
        response = result.replace(prompt.replace(tokenizer.eos_token, ' ').replace(tokenizer.bos_token, ''), '')
        print(response)
        print('=====END=====')
        return response


# %%
conversation = [
    {
        "role": "function_metadata",
        "content": f"```\n{functions}\n```\n{instruction}"
    },
    {
        "role": "user",
        # "content": "who is the current president of bangladesh?"
        # 'content': "compare nutron star and banana"
        # 'content': "1. who is the current president of bangladesh\n2. who was the first president of bangladesh"
        # 'content': 'write an essay on globalization'
        # 'content': 'write a poem on oscars'
        # 'content': 'The distance between towns A and B is 300 km. One train departs from town A and another train departs from town B, both leaving at the same moment of time and heading towards each other. We know that one of them is 10 km/hr faster than the other. Find the speeds of both trains if 2 hours after their departure the distance between them is 40 km.'
        # 'content': 'what capabilities do you have?'
        # 'content': 'hi there'
        # "content": 'what is 02348.2 / 56.345'
        "content": "Elmer Fudd decided to grow a garden so he could make salad. He wants to make it 10.1 m long and 4.2 m wide. However, in order to avoid Bugs Bunny from entering his garden he must make a fence surrounding the garden. He decides to make the fence 11.2 m long and 5.0 m wide. What is the area between the fence and the garden?"
    }
]
response_str = generate(conversation)


#%%
while True:
    response_str = generate(conversation)
    if '[FUNCTION_CALL]' not in response_str:
        break
    tool = json.loads(response_str.replace('[FUNCTION_CALL]', '').replace('[/FUNCTION_CALL]', '').strip())
    search_results = []
    python_results = []
    
    if tool['name'] == 'get_web_search_result':
        query = tool['arguments']['query']
        bing_results = bing.query(query)
        bing_results = [asdict(result) for result in bing_results]
        search_results.extend(bing_results)
    if tool['name'] == 'get_python_math_result':
        expression = tool['arguments']['expression']
        try:
            result = {}
            exec(expression, result)
            python_results.append("Python answer = " + str(result['result']))
        except Exception as e:
            python_results.append("Python answer = " + traceback.format_exc())

    result = ''
    if len(search_results) > 0:
        result += json.dumps(search_results, indent=2) + '\n'
    if len(python_results) > 0:
        result += '\n'.join(python_results) + '\n'
    result = result.strip()
    conversation.extend([
        {
            'role': 'function_call',
            'content': response_str.replace('[FUNCTION_CALL]', '').replace('  ', ' ')
        },
        {
            'role': 'function_response',
            'content': result
        }
    ])

