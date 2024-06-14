
from label_studio_sdk import Client
import json

class LabelStudio:
    def __init__(
        self, url: str, api_key: str, single_turn_web_search_project_id: int, multi_turn_web_search_project_id: int
    ):
        ls = Client(url=url, api_key=api_key)
        ls.check_connection()
        self.web_search_single_project = ls.get_project(single_turn_web_search_project_id)
        self.web_search_multi_project = ls.get_project(multi_turn_web_search_project_id)

    def log_single_turn(
            self,
            user_message: str,
            tool_name: str,
            argument_label: str,
            argument_values: list[str]
    ):
        tools = []
        for arg in argument_values:
            tools.append({
                "name": tool_name,
                "arguments": {
                    argument_label: arg
                }
            })
        prediction = {
            "tools": tools
        }
        prediction = json.dumps(prediction, indent=2)
        data = { "data": {
            "user_1": user_message,
            "category": "manual"
        }, "predictions": [{
            "result": [
                {
                    "value": {"text": [prediction]},
                    "from_name": "function_1",
                    "to_name": "user_1",
                    "type": "textarea",
                    "origin": "manual",
                }
            ]
        }]}
        result = self.web_search_single_project.import_tasks(
            data
        )
        return isinstance(result, list) and len(result) == 1

    def log_single_turn_web_search(
        self,
        id: str,
        row
    ):
        search_term = row['search_terms']['q1_search_term']
        row['row_id'] = id
        row['user_1'] = row['conversation'][0]['content']
        prediction = {
            "tools": [{
                "name": 'get_web_search_result',
                "arguments": {
                    'query': search_term
                }
            }]
        }
        prediction = json.dumps(prediction, indent=2)
        data = { "data": row, "predictions": [{
            "result": [
                {
                    "value": {"text": [prediction]},
                    "from_name": "function_1",
                    "to_name": "user_1",
                    "type": "textarea",
                    "origin": "manual",
                }
            ]
                
        }]}
        result = self.web_search_single_project.import_tasks(
            data
        )
        return isinstance(result, list) and len(result) == 1

    def log_multi_turn_web_search(self, data, assistant_response, function_prediction):
        result = self.web_search_multi_project.import_tasks({
            "data": data,
            "predictions": [{"result": [{
                "value": {"text": [assistant_response]},
                "from_name": "assistant_1",
                "to_name": "table",
                "type": "textarea",
                "origin": "manual",
            }, {
                "value": {"text": [function_prediction]},
                "from_name": "function_call_3",
                "to_name": "user_2",
                "type": "textarea",
                "origin": "manual",
            }]}]
        })
        return isinstance(result, list) and len(result) == 1