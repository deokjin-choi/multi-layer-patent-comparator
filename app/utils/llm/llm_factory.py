# app/utils/llm/llm_factory.py

import os

def get_llm_client():
    """
    환경변수 LLM_MODE 값에 따라 알맞은 LLM client 인스턴스를 반환합니다.
    - LLM_MODE=local   → 로컬 Ollama API 방식
    - LLM_MODE=gauss   → LangChain 기반 사내 가우스 API
    """
    mode = os.getenv("LLM_MODE", "local").lower()

    if mode == "gauss":
        from .gauss_llm_client import GaussLLMClient
        return GaussLLMClient()
    else:
        from .local_api_llm_client import LocalAPILLMClient
        return LocalAPILLMClient()
