import os
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
load_dotenv()

#%%
HF_TOKEN = os.getenv("HF_TOKEN")
model_folder = "./outputs/output_models_lr_0.000005-v3-only-raquib-annotation/checkpoint-3270"
chat_template = open("prompts/template.txt").read()
HF_repo = "MerlynMind/Function-Calling-v0.2-Mistral-7B-v0.1"

#%%
config = PeftConfig.from_pretrained(f"{model_folder}/adapter_model")
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path, device_map="auto"
)
inference_model = PeftModel.from_pretrained(model, f"{model_folder}/adapter_model")
inference_model = inference_model.merge_and_unload()
tokenizer = AutoTokenizer.from_pretrained(
    config.base_model_name_or_path, padding_side="right", use_fast=True
)
tokenizer.chat_template = chat_template
inference_model = inference_model.half()

# %%
tokenizer.push_to_hub(repo_id=HF_repo, token=HF_TOKEN)
inference_model.push_to_hub(repo_id=HF_repo, token=HF_TOKEN)
