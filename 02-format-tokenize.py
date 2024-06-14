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
import random

# %%
dataset_id = 'MerlynMind/ATG_Function_Call_SFT_V1'
base_model_id = "mistralai/Mistral-7B-v0.1"
instruction = open("prompts/instruction.txt", "r").read()
chat_template = open("prompts/template.txt", "r").read()

# %%
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id, model_max_length=8194, padding_side="left", add_eos_token=True
)
tokenizer.chat_template = chat_template
tokenizer.pad_token = tokenizer.eos_token


def generate_and_tokenize_prompt(functions, data_point):
    conversation = [
        {
            "role": "function_metadata",
            "content": f"```\n{json.dumps(functions)}\n```\n{instruction}",
        }
    ]
    conversation.extend(data_point["conversation"])
    return {
        "input": tokenizer.apply_chat_template(conversation[:-1], tokenize=False),
        "output": tokenizer.apply_chat_template(conversation[-1:], tokenize=False)[3:],
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
with open("datasets/dataset-v2.jsonl", "r") as f:
    lines = f.readlines()
    dataset = [json.loads(line) for line in lines]

#%%
cateogry_weight_map = {
    'manual': 1,
    'MerlynMind/RAG_Current_Events_v1_20240220': 1,
    'chat_alpaca': 1,
    'lilacai/glaive-function-calling-v2-sharegpt': 0.1
}

new_train_dataset = []
new_test_dataset = []
for category in cateogry_weight_map.keys():
    weight = cateogry_weight_map[category]
    if weight == 1:
        new_train_dataset.extend([row for row in dataset if row['source_id'].split(':')[0] == category])
        continue
    category_data = [row for row in dataset if row['source_id'].split(':')[0] == category]
    category_data = [row for row in category_data if any([message['role'] == 'function_call' for message in row['conversation']])]
    random.shuffle(category_data)
    sample_size = int(len(category_data) * weight)
    sampled_data = random.sample(category_data, sample_size)
    test_size = int(sample_size * 0.2)
    new_train_dataset.extend(sampled_data[test_size:])
    new_test_dataset.extend(sampled_data[:test_size])
dataset = []


# %%
ds = {
    'train': [],
    'test': []
}
input_token_lengths = {
    'train': [],
    'test': []
}
output_token_lengths = {
    'train': [],
    'test': []
}
total_token_lengths = {
    'train': [],
    'test': []
}
for split, rows in zip(['train', 'test'], [new_train_dataset, new_test_dataset]):
    for row in tqdm(rows):
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
            new_row = generate_and_tokenize_prompt(row['functions'], new_row)
            ds[split].append(new_row)
            input_token_length = len(tokenizer(new_row['input'])['input_ids'])
            output_token_length = len(tokenizer(new_row['output'])['input_ids'])
            input_token_lengths[split].append(input_token_length)
            output_token_lengths[split].append(output_token_length)
            total_token_lengths[split].append(input_token_length + output_token_length)

#%%
# write train and test datasets to jsonl file
for split in ['train', 'test']:
    with open(f"datasets/{split}-formatted-dataset.jsonl", "w") as f:
        for row in ds[split]:
            f.write(json.dumps(row) + "\n")

#%%
# plot token length distributions in subplots
# fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# axs[0].hist(input_token_lengths, bins=100)
# axs[0].set_title("Input token length")
# axs[0].set_xlabel("Token length")
# axs[0].set_ylabel("Frequency")
# axs[1].hist(output_token_lengths, bins=100)
# axs[1].set_title("Output token length")
# axs[1].set_xlabel("Token length")
# axs[1].set_ylabel("Frequency")
# axs[2].hist(total_token_lengths, bins=100)
# axs[2].set_title("Total token length")
# axs[2].set_xlabel("Token length")
# axs[2].set_ylabel("Frequency")
# plt.tight_layout()
# plt.show()

# %%
# with open("datasets/formatted-dataset.jsonl", "r") as f:
#     lines = f.readlines()
#     ds = [json.loads(line) for line in lines]
