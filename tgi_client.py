import json
from typing import Callable

import requests
from dataclasses import dataclass

@dataclass
class ConnectionMetadata:
    connection_id: str
    response_message_id: int

class TgiClient:
    def __init__(self, url: str, temperature: float = 0.001, max_new_tokens: int = 4096):
        self.url = url
        self.params = {
            "do_sample": False,
            "top_p": 0.999,
            "temperature": 0.001,
            "max_new_tokens": 4096,
            "top_k": 50,
            "stop": ["</s>"],
            "repetition_penalty": 1.0,
        }

    def generate(self, text: str) -> str:
        url = self.url + "/generate"
        headers = {"Content-Type": "application/json"}
        data = {"inputs": text, "parameters": self.params}
        response = requests.post(url, headers=headers, json=data, stream=False)
        response = response.json()["generated_text"]
        return response

    def generate_stream(
        self,
        text: str,
        connection_metadata: ConnectionMetadata,
        callback: Callable[[str, bool, ConnectionMetadata], None],
    ):
        url = self.url + "/generate_stream"
        headers = {"Content-Type": "application/json"}
        body = {"inputs": text, "parameters": self.params}
        response = requests.post(url, headers=headers, json=body, stream=True)
        response_str = ""
        for chunk in response.iter_content(chunk_size=4095):
            if chunk:
                chunk = chunk.decode("utf-8")[5:]
                try:
                    json_payload = json.loads(chunk)
                    is_final = json_payload["generated_text"] is not None
                    if is_final:
                        response_str = json_payload["generated_text"]
                    else:
                        response_str = response_str + json_payload["token"]["text"]
                    callback(response_str, is_final, connection_metadata)
                except Exception:
                    pass

        response.close()