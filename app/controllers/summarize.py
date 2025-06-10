from app.utils.llm.llm_factory import get_llm_client
from app.utils.prompts import load_prompt
from app.utils.json_helper import extract_json_from_llm_output  # 위 함수 별도 분리 시
from app.utils.llm.retry_utils import safe_invoke


def summarize_patent(text: str, patent_id: str, prompt_version: str = "v1") -> dict:
    llm = get_llm_client()
    
    prompt_template = load_prompt("summarize", prompt_version)
    filled_prompt = prompt_template.replace("{{description}}", text[:4000])  # LLM 입력 길이 제한 고려

    response = safe_invoke(llm, filled_prompt, extract_json_from_llm_output)  # 재시도 포함
    
    if response is None:
        response = {
            "problem": "N/A",
            "solution_function": "N/A",
            "solution_structure": "N/A",
            "solution_implementation": "N/A",
            "effect": "N/A"
        }

    response['id'] = patent_id  # patent_id 추가  
    
    return response
