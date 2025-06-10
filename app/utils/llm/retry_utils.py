# app/utils/llm/retry_utils.py

import time
from app.utils.json_helper import extract_json_from_llm_output  # 위 함수 별도 분리 시

def safe_invoke(llm, prompt: str, parse_func=extract_json_from_llm_output, max_retries=3, delay=1):
    """
    LLM 호출 후 파싱까지 포함해서 재시도. 파싱이 실패해도 재시도함.
    
    Args:
        llm: LLM client
        prompt: str, 프롬프트 텍스트
        parse_func: LLM 응답을 JSON으로 바꾸는 함수 (예: extract_json_from_llm_output)
    
    Returns:
        parsing 결과 또는 None
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = llm.invoke(prompt)
            parsed = parse_func(response)
            return parsed  # 성공 시 바로 리턴
        except Exception as e:
            print(f"[safe_invoke] 시도 {attempt}/{max_retries} 실패: {e}")
            time.sleep(delay)
    return None
