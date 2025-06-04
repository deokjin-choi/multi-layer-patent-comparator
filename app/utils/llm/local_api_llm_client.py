# app/utils/llm/local_api_llm_client.py

import requests
from .base_llm_client import BaseLLMClient

class LocalAPILLMClient(BaseLLMClient):
    """
    로컬에서 실행 중인 Ollama 등 설치형 LLM 서버에 REST API로 요청을 보내는 클라이언트입니다.
    """

    def __init__(self, model: str = "mistral", endpoint: str = "http://localhost:11434/api/generate"):
        self.model = model
        self.endpoint = endpoint

    def invoke(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.2
        }

        try:
            response = requests.post(self.endpoint, json=payload)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            raise RuntimeError(f"[LocalAPILLMClient] LLM 호출 중 오류 발생: {e}")
