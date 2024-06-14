#%%
import json
from dataclasses import asdict
import traceback
from bing_client import BingClient
from dataclasses import dataclass
from tgi_client import TgiClient, ConnectionMetadata
from typing import Callable

#%%
@dataclass
class Message:
    role: str
    content: int

class Orchestrator:
    def __init__(self, bing_key: str, tgi_url: str):
        self.bing = BingClient(bing_key)
        self.functions = open("prompts/functions.txt", "r").read()
        self.instruction = open("prompts/instruction.txt", "r").read()
        self.chat_template = open("prompts/template.txt", "r").read()
        self.tgi_client = TgiClient(tgi_url)

    def apply_chat_template(self, conversation: list[Message]):
        bos_token = "<s>"
        eos_token = "</s>"
        formatted_conversation = bos_token
        conversation.insert(0, Message('function_metadata', f"```\n{self.functions}\n```\n{self.instruction}"))
        for message in conversation:
            if message.role == 'function_metadata':
                formatted_conversation += f"[INST] You have access to the following functions. Use them if required:\n{message.content} [/INST]Sure. I will follow these instructions.{eos_token}"
            elif message.role == 'user':
                formatted_conversation += f"[INST] {message.content} [/INST]"
            elif message.role == 'assistant':
                formatted_conversation += f"{message.content}{eos_token}"
            elif message.role == 'function_call':
                formatted_conversation += f"[FUNCTION_CALL] {message.content}{eos_token}"
            elif message.role == 'function_response':
                formatted_conversation += f"[FUNCTION_RESPONSE] Here is the response to the function call. If helpful, use it to respond to the user's question: {message.content} [/FUNCTION_RESPONSE]"
        return formatted_conversation

    def query(self, conversation: list[Message], connectionMetadata: ConnectionMetadata, callback: Callable[[ConnectionMetadata, str], None]):
        response_str = "NO RESPONSE"
        while True:
            formatted_conversation = self.apply_chat_template(conversation)
            response_str = self.tgi_client.generate(formatted_conversation)
            if '[FUNCTION_CALL]' not in response_str:
                break
            tool = json.loads(response_str.replace('[FUNCTION_CALL]', '').replace('[/FUNCTION_CALL]', '').strip())
            search_results = []
            python_results = []
            
            if tool['name'] == 'get_web_search_result':
                callback(connectionMetadata, "Performing web search")
                query = tool['arguments']['query']
                bing_results = self.bing.query(query)
                bing_results = [asdict(result) for result in bing_results]
                search_results.extend(bing_results)
            if tool['name'] == 'get_python_math_result':
                callback(connectionMetadata, "Evaluating math calculation")
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
                Message('function_call', response_str.replace('[FUNCTION_CALL]', '').replace('  ', ' ')),
                Message('function_response', result)
            ])
        callback(connectionMetadata, response_str)