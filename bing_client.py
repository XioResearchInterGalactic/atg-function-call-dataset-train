from dataclasses import dataclass

import requests
from dacite import from_dict

import time

from typing import Callable, TypeVar

@dataclass
class BingResult:
    snippet: str
    url: str


T = TypeVar("T")

class BingClient:
    def __init__(self, bing_key: str):
        self.url = "https://api.bing.microsoft.com/v7.0/search"
        self.key = bing_key


    def retry_with_exponential_delay(
        self,
        callback: Callable[[], T],
        n: int = 10,
        base_delay_ms: int = 1000,
        max_delay_ms: int = 60000,
    ):
        for attempt in range(1, n + 1):
            try:
                result = callback()
                return result
            except Exception as e:
                print(f"Attempt {attempt} failed: {e}")
                if attempt < n:
                    delay = min(
                        (base_delay_ms / 1000) * (2 ** (attempt - 1)), (max_delay_ms / 1000)
                    )
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)

        return None
    def __make_bing_request(
        self, headers: dict[str, str], params: dict[str, str | bool]
    ):
        response = requests.get(self.url, headers=headers, params=params)
        response.raise_for_status()
        return response

    def query(self, search_term: str) -> list[BingResult] | None:
        # Call bing API
        headers = {"Ocp-Apim-Subscription-Key": self.key}
        params: dict[str, str | bool] = {
            "q": search_term,
            "textDecorations": True,
            "textFormat": "Raw",
        }
        response = self.retry_with_exponential_delay(
            lambda: self.__make_bing_request(headers, params)
        )

        if response is None:
            return None

        # Extract results
        try:
            results = response.json()["webPages"]["value"]
            return self.cleanup_information(
                [from_dict(BingResult, row) for row in results]
            )
        except Exception:
            return None
        
    
    def cleanup_information(self, bing_results: list[BingResult]) -> list[BingResult]:
        results: list[BingResult] = []
        for result in bing_results:
            result.snippet = result.snippet.replace("", "").replace("", "")
            results.append(result)
        return results