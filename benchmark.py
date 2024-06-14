#%% IMPORT
import json
from transformers import AutoTokenizer
from tqdm import tqdm
from tgi_client import TgiClient
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import os
import io
import sys
import traceback
import pandas as pd
load_dotenv()

#%% CONSTANTS
base_model_id = "mistralai/Mistral-7B-v0.1"
instruction = open("prompts/instruction.txt", "r").read()
chat_template = open("prompts/template.txt", "r").read()
hf_token = os.getenv("HF_TOKEN")

#%% LOAD VALIDATION DATA
ds = json.load(open('datasets/validation-dataset.json'))

#%% KEEP ANNOTATED ONLY
print(f"Initial dataset size: {len(ds)}")
ds2 = []
for row in ds:
    if row['category'] != 'lilacai/glaive-function-calling-v2-sharegpt':
        ds2.append(row)
ds = ds2
print(f"Filtered dataset size: {len(ds)}")

#%% TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id, model_max_length=8194, padding_side="left", add_eos_token=True, token=hf_token
)
tokenizer.chat_template = chat_template
tokenizer.pad_token = tokenizer.eos_token

#%% FUNCTIONS
def generate_and_tokenize_prompt(functions, conversation):
    new_conversation = [
        {
            "role": "function_metadata",
            "content": f"```\n{json.dumps(functions)}\n```\n{instruction}",
        }
    ]
    new_conversation.extend(conversation)
    return {
        "input": tokenizer.apply_chat_template(new_conversation[:-1], tokenize=False),
        "output": tokenizer.apply_chat_template(new_conversation[-1:], tokenize=False)[3:].strip(),
    }

def extract_function_call(text: str) -> dict[str, any] | None:
    try:
        index = text.index("{")
        index2 = text.rindex("}")
        return json.loads(text[index:index2+1])
    except Exception:
        return None
    
#%% FORMAT PROMPTS
ds2 = []
for row in ds:
    indices = [
        i
        for i, x in enumerate(row["conversation"])
        if x["role"] in ["function_call", "assistant"]
    ]
    for index in indices:
        new_row = generate_and_tokenize_prompt(json.loads(row['functions']), row['conversation'][: index + 1])
        ds2.append(new_row)
ds = ds2

#%% RUN INFERENCE
ds2 = []
tgi_client = TgiClient("http://localhost:8081", max_new_tokens=2048)
for row in tqdm(ds, desc="Running inference"):
    try:
        response = tgi_client.generate(row['input'])
        row['generated'] = response.strip()
        ds2.append(row)
    except Exception as e:
        print(f"Error: {e}")
ds = ds2
pd.DataFrame(ds).to_csv("inferences.csv", index=False)

#%% PYTHON RUNNER
def exec_python(snippet: str):
    try:
        result: dict[str, str] = {}
        buffer = io.StringIO()
        sys.stdout = buffer
        exec(snippet, result)
        sys.stdout = sys.__stdout__
        result.pop("__builtins__", None)
        for key in result:
            result[key] = str(result[key])[:40]
        printed_output = buffer.getvalue()
        result = {
            'variables': json.dumps(result),
            'printed_output': printed_output
        }
    except Exception as e:
        result = {"error": traceback.format_exc()}
    return result

#%% GENERATE METRICS
ds = pd.read_csv("inferences.csv").to_dict(orient="records")
model = SentenceTransformer("all-MiniLM-L6-v2")
metrics = []
for row in tqdm(ds, desc="Calculating metrics"):
    type_correct = False
    is_python_expected = False
    python_runnable = False
    json_correct = False

    # CHECK IF EMPTY
    if len(str(row['generated']).strip()) == 0 or str(row['generated']) == 'nan':
        metrics.append({
            "json_correct": False,
            "type_correct": False,
            "semantic_similarity": 0.0,
            "is_python_expected": False,
            "python_runnable": False
        })
        continue

    # CHECK JSON CORRECT
    if row['generated'].startswith("[FUNCTION_CALL]"):
        json_correct = False
        if extract_function_call(row['generated']) is not None:
            json_correct = True
    else:
        json_correct = True

    # CHECK IF TYPE IS CORRECT
    if row['output'].startswith("[FUNCTION_CALL]"):
        if row['generated'].startswith("[FUNCTION_CALL]"):
            generated_function = extract_function_call(row['generated'])
            expected_function = extract_function_call(row['output'])
            if generated_function is not None and expected_function is not None and generated_function['name'] == expected_function['name']:
                type_correct = True

                if expected_function['name'] == 'get_python_math_result':
                    is_python_expected = True
                    try:
                        code = generated_function['arguments']['expression'].strip()
                        result = exec_python(code)
                        if 'error' not in result and 'printed_output' in result:
                            python_runnable = True
                    except Exception as e:
                        print(f"Error: {e}")
                        pass
    elif "[FUNCTION_CALL]" not in row['generated'] and '[FUNCTION_CALL]' not in row['output']:
        type_correct = True

    # CALCULATE SEMANTIC DISTANCE
    expected_embeddings = model.encode(row['output'], convert_to_tensor=True)
    generated_embeddings = model.encode(row['generated'], convert_to_tensor=True)
    cosine_scores = util.cos_sim(expected_embeddings, generated_embeddings)
    metrics.append({
        "json_correct": json_correct,
        "type_correct": type_correct,
        "semantic_similarity": cosine_scores[0][0],
        "is_python_expected": is_python_expected,
        "python_runnable": python_runnable
    })

#%% AGGREGATE METRICS
percent_json_correct = sum([1 for x in metrics if x['json_correct']]) / len(metrics)
percent_type_correct = sum([1 for x in metrics if x['type_correct']]) / len(metrics)
average_semantic_similarity = sum([x['semantic_similarity'] for x in metrics]) / len(metrics)
max_semantic_similarity = max([x['semantic_similarity'] for x in metrics])
min_semantic_similarity = min([x['semantic_similarity'] for x in metrics])
python_expected_count = sum([1 for x in metrics if x['is_python_expected']])
python_runnable_count = sum([1 for x in metrics if x['python_runnable']])
python_accuracy = python_runnable_count / python_expected_count
results = {
    "percent_json_correct": percent_json_correct,
    "percent_type_correct": percent_type_correct,
    "average_semantic_similarity": average_semantic_similarity,
    "max_semantic_similarity": max_semantic_similarity,
    "min_semantic_similarity": min_semantic_similarity,
    'python_accuracy': python_accuracy,
}
for key, value in results.items():
    print(f"{key}: {value}")
