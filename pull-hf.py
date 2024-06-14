import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from string import Template

ft_model_id = "MerlynMind/Function-Calling-v0.2-Mistral-7B-v0.1"
hf_token = "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"
tokenizer = AutoTokenizer.from_pretrained(
    ft_model_id, add_bos_token=True, trust_remote_code=True, token=hf_token
)
model = AutoModelForCausalLM.from_pretrained(
    ft_model_id, device_map="auto", trust_remote_code=True, token=hf_token
)
model.eval()

# Prompt
instruction = """
<s>[INST] You have access to the following functions. Use them if required:
```
[{"type": "object", "properties": {"name": {"type": "string", "value": "get_web_search_result", "description": "Search the web to fetch up to date information"}, "arguments": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query to find information"}}, "required": ["query"]}}, "required": ["name", "arguments"]}, {"type": "object", "properties": {"name": {"type": "string", "value": "get_python_math_result", "description": "Perform mathematical or date operations using python programming language."}, "arguments": {"type": "object", "properties": {"expression": {"type": "string", "description": "Python script that performs mathematical or date operations to calculate results. The final result must be printed using `print` function."}}, "required": ["expression"]}}, "required": ["name", "arguments"]}]
```
You are a helpful AI assistant named Merlyn AI made by Merlyn Mind. Your task is to answer users questions. To answer it, you have the above functions at your disposal. You can invoke the functions and the function responses will be provided back to you that will help you answer user's question. You must follow these instructions:
- If you don't know the answer to the user's question, then select one or more of the above functions based on the user query
- If a function is found, you must respond a FUNCTION_CALL tag in the JSON format matching the following the functions schema mentioned above.
- Whenever you respond with FUNCTION_CALL, a corresponding FUNCTION_RESPONSE will be provided back to you.
- If there are multiple functions required, you can call them one after another, each within its FUNCTION_CALL block
- If there is no function that match the user request, you will respond as you normally would without doing a function call.
- Do not add any additional Notes or Explanations for FUNCTION_CALL. [/INST]Sure. I will follow these instructions.</s>[INST] ${question} [/INST]
""".strip()
conversation = Template(instruction).substitute(question="What is the capital of France?")

model_inputs = tokenizer(conversation, return_tensors="pt", add_special_tokens=False).to("cuda")
with torch.no_grad():
    result = model.generate(**model_inputs, max_new_tokens=4096)[0]
    result = tokenizer.decode(result, skip_special_tokens=True)
    print(result)
