# tests/test_llm_call.py

import os
import sys
import importlib
from app.utils.llm.retry_utils import safe_invoke

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 📌 reload 대상 모듈 import
import app.utils.llm.local_api_llm_client as local_llm
import app.utils.llm.llm_factory as llm_factory

# 📌 수정 사항 즉시 반영
importlib.reload(local_llm)
importlib.reload(llm_factory)

def test_llm():
    os.environ["LLM_MODE"] = "local"
    
    # 📌 llm_factory는 reload 되었기 때문에 최신 local_llm을 반영
    llm = llm_factory.get_llm_client()
    response = llm.invoke("Briefly explain about Petronas Groupo")
    print(response)

if __name__ == "__main__":
    test_llm()
