# app/utils/json_helper.py

import json
import re

def extract_json_from_llm_output(text_or_obj) -> dict:
    """
    Parses JSON from LLM response, safely handling both raw string and dict.
    Removes markdown-style code fences and guarantees clean JSON result.

    Args:
        text_or_obj (str or dict): LLM response as string or already-parsed dict

    Returns:
        dict: Parsed JSON dict only. If failed, returns {'error': ..., 'raw_output': ...}
    """
    # 1. 이미 dict면 그대로 반환 (추가 key 없이!)
    if isinstance(text_or_obj, dict):
        return text_or_obj

    # 2. 문자열 처리
    text = str(text_or_obj).strip()
    cleaned = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.IGNORECASE | re.MULTILINE)

    try:
        return json.loads(cleaned)
    except Exception:
        # → 여기서 예외를 던져야 safe_invoke가 재시도함
        raise ValueError("Failed to parse JSON: " + text[:200])  # 일부만 로그로
