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

# %%
base_model_id = "allyson-ai/FuncMaster-v0.1-Mistral-7B"
functions = json.dumps(json.load(open("prompts-func-master/functions-v2.json", "r")))
instruction = open("prompts-func-master/instruction.txt", "r").read()
# chat_template = open("prompts-func-master/template.txt", "r").read()

# %%
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id, model_max_length=8194, padding_side="left", add_eos_token=True
)
tokenizer.pad_token = tokenizer.eos_token

def generate_and_tokenize_prompt(data_point):
    conversation = [
        {
            "from": "system",
            "value": Template(instruction).substitute(functions=functions),
        }
    ]
    for row in data_point["conversation"]:
        if row["role"] == "function_call":
            value = f"<functioncall> {row['content']} <|endoftext|>"
        elif row["role"] == "function_response":
            value = f"Function Response: {row['content']}"
        else:
            value = row["content"]
        conversation.append(
            {
                "from": "user"
                if row["role"] == "user" or row["role"] == "function_response"
                else "gpt",
                "value": value,
            }
        )
    print(json.dumps(conversation))
    return {
        "input": tokenizer.apply_chat_template(conversation[:-1], tokenize=False),
        "output": tokenizer.apply_chat_template(conversation[-1:], tokenize=False),
    }


def function_json(message):
    if message["role"] == "function_call":
        function_call = json.loads(message["content"])
        if "tools" in function_call:
            tool = function_call["tools"][0]
        else:
            tool = function_call
        message["content"] = json.dumps(tool, indent=2)
    return message


# %%
# read train dataset from jsonl to list
with open("datasets/dataset.jsonl", "r") as f:
    lines = f.readlines()
    train_dataset = [json.loads(line) for line in lines]

# %%
ds = []
input_token_lengths = []
output_token_lengths = []
total_token_lengths = []
for row in tqdm(train_dataset):
    indices = [
        i
        for i, x in enumerate(row["conversation"])
        if x["role"] in ["function_call", "assistant"]
    ]
    for index in indices:
        new_row = row.copy()
        new_row["conversation"] = row["conversation"][: index + 1]
        new_row["conversation"] = [
            function_json(message) for message in new_row["conversation"]
        ]
        new_row = generate_and_tokenize_prompt(new_row)
        # print(new_row)
        ds.append(new_row)
        input_token_length = len(tokenizer(new_row["input"])["input_ids"])
        output_token_length = len(tokenizer(new_row["output"])["input_ids"])
        input_token_lengths.append(input_token_length)
        output_token_lengths.append(output_token_length)
        total_token_lengths.append(input_token_length + output_token_length)

# %%
# plot token length distributions in subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].hist(input_token_lengths, bins=100)
axs[0].set_title("Input token length")
axs[0].set_xlabel("Token length")
axs[0].set_ylabel("Frequency")
axs[1].hist(output_token_lengths, bins=100)
axs[1].set_title("Output token length")
axs[1].set_xlabel("Token length")
axs[1].set_ylabel("Frequency")
axs[2].hist(total_token_lengths, bins=100)
axs[2].set_title("Total token length")
axs[2].set_xlabel("Token length")
axs[2].set_ylabel("Frequency")
plt.tight_layout()
plt.show()

# %%
with open("datasets/formatted-dataset-func-master.json", "w") as f:
    json.dump(ds, f, indent=2)


# %%
