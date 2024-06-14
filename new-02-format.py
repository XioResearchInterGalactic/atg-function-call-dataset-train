# %%
from transformers import AutoTokenizer
import json
from datasets import Dataset, load_dataset, DatasetDict
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from dotenv import load_dotenv
import os

# %%
load_dotenv()
dataset_id = 'MerlynMind/ATG_Function_Call_SFT_V1'
target_dataset_id = 'MerlynMind/ATG_Function_Call_Formatted_Mistral_7B_v0.1_SFT_V1'
base_model_id = "mistralai/Mistral-7B-v0.1"
instruction = open("prompts/instruction.txt", "r").read()
chat_template = open("prompts/template.txt", "r").read()
hf_token = os.getenv("HF_TOKEN")
max_input_length = 6144
max_output_length = 2048

#%%
train_dataset_path = 'datasets/train-dataset.json'
validation_dataset_path = 'datasets/validation-dataset.json'

# %%
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id, model_max_length=8194, padding_side="left", add_eos_token=True
)
tokenizer.chat_template = chat_template
tokenizer.pad_token = tokenizer.eos_token

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
        "output": tokenizer.apply_chat_template(new_conversation[-1:], tokenize=False)[3:],
    }

def plot_token_length_distribution(lengths_input, lengths_output, lengths_total, title):
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    axs[0].hist(lengths_input, bins=100)
    axs[0].set_title("input token length")
    axs[0].set_xlabel("Token length")
    axs[0].set_ylabel("Frequency")
    axs[1].hist(lengths_output, bins=100)
    axs[1].set_title("output token length")
    axs[1].set_xlabel("Token length")
    axs[1].set_ylabel("Frequency")
    axs[2].hist(lengths_total, bins=100)
    axs[2].set_title("total token length")
    axs[2].set_xlabel("Token length")
    axs[2].set_ylabel("Frequency")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

#%%
dataset = load_dataset(dataset_id, token=hf_token)
train_dataset = dataset['train']
validation_dataset = dataset['validation']
print(len(train_dataset))
print(len(validation_dataset))


#%% LOAD DATASET FROM JSON FILE
train_dataset = json.load(open(train_dataset_path, "r"))
validation_dataset = json.load(open(validation_dataset_path, "r"))

#%%
new_rows = {
    'train': [],
    'validation': []
}
for split, rows in zip(['train', 'validation'], [train_dataset, validation_dataset]):
    for row in tqdm(rows):
        indices = [
            i
            for i, x in enumerate(row["conversation"])
            if x["role"] in ["function_call", "assistant"]
        ]
        for index in indices:
            new_row = row.copy()
            new_row = generate_and_tokenize_prompt(json.loads(row['functions']), new_row['conversation'][: index + 1])
            input_token_length = len(tokenizer(new_row['input'])['input_ids'])
            output_token_length = len(tokenizer(new_row['output'])['input_ids'])
            new_rows[split].append(new_row)

#%%
def plot_lengths(train_dataset, validation_dataset):
    lengths = {
        "train_input": [],
        'train_output': [],
        "train_total": [],
        "validation_input": [],
        'validation_output': [],
        "validation_total": [],
    }
    for split, rows in zip(['train', 'validation'], [train_dataset, validation_dataset]):
        for row in tqdm(rows):
            input_token_length = len(tokenizer(row["input"])["input_ids"])
            output_token_length = len(tokenizer(row["output"])["input_ids"])
            lengths[split + "_input"].append(input_token_length)
            lengths[split + "_output"].append(output_token_length)
            lengths[split + "_total"].append(input_token_length + output_token_length)
                
    plot_token_length_distribution(lengths["train_input"], lengths['train_output'], lengths["train_total"], "Train total token length")
    plot_token_length_distribution(lengths["validation_input"], lengths['validation_output'], lengths["validation_total"], "Validation total token length")
    print("Max train total token length:", max(lengths["train_total"]))
    print("Max train output token length:", max(lengths["train_output"]))
    print("Max train input token length:", max(lengths["train_input"]))
    print("Max validation total token length:", max(lengths["validation_total"]))
    print("Max validation output token length:", max(lengths["validation_output"]))
    print("Max validation input token length:", max(lengths["validation_input"]))

#%% SHUFFLE ROWS
new_rows["train"] = random.sample(new_rows["train"], len(new_rows["train"]))
new_rows["validation"] = random.sample(new_rows["validation"], len(new_rows["validation"]))

#%% REMOVE ROWS WITH LENGTH GREATER THAN MAX
plot_lengths(new_rows['train'], new_rows['validation'])
print("Number of train examples before filtering:", len(new_rows["train"]))
print("Number of validation examples before filtering:", len(new_rows["validation"]))
new_rows["train"] = [row for row in new_rows["train"] if len(tokenizer(row["input"])['input_ids']) <= max_input_length]
new_rows['train'] = [row for row in new_rows['train'] if len(tokenizer(row['output'])['input_ids']) <= max_output_length]
new_rows["validation"] = [row for row in new_rows["validation"] if len(tokenizer(row["input"])['input_ids']) <= max_input_length]
new_rows['validation'] = [row for row in new_rows['validation'] if len(tokenizer(row['output'])['input_ids']) <= max_output_length]
print("Number of train examples after filtering:", len(new_rows["train"]))
print("Number of validation examples after filtering:", len(new_rows["validation"]))
plot_lengths(new_rows['train'], new_rows['validation'])

#%% UPLOAD TO HF
train_dataset = Dataset.from_list(new_rows['train'])
validation_dataset = Dataset.from_list(new_rows['validation'])
dataset_dict = DatasetDict({"train": train_dataset, "validation": validation_dataset})
dataset_dict.push_to_hub(target_dataset_id, token=hf_token)

#%% write train and test datasets to jsonl file
for split in ['train', 'validation']:
    with open(f"datasets/{split}-formatted-dataset.jsonl", "w") as f:
        for row in new_rows[split]:
            f.write(json.dumps(row) + "\n")

# %%
