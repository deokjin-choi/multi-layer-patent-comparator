import pandas as pd
import json
from app.utils.llm.llm_factory import get_llm_client
from app.utils.prompts import load_prompt
from app.utils.json_helper import extract_json_from_llm_output  # 위 함수 별도 분리 시


def analyze_implementation_diff(our_patent: dict, competitor_patents: list[dict]) -> list[pd.DataFrame]:
    """
    Implementation 관점에서 당사 특허와 경쟁사 특허 간의 차이점을 비교하여
    DataFrame 리스트로 반환합니다.
    """

    llm = get_llm_client()
    prompt_template = load_prompt("implementation", "diff_v8")

    # 당사 특허 ID와 summary 분리
    our_id = our_patent["id"]
    our_summary = {k: v for k, v in our_patent.items() if k != "id"}

    comparison_tables = []

    for comp in competitor_patents:
        comp_id = comp["id"]
        comp_summary = {k: v for k, v in comp.items() if k != "id"}

        # 프롬프트 완성
        filled_prompt = prompt_template.format(
            our_problem=our_summary.get("problem", ""),
            our_function=our_summary.get("function", ""),
            our_structure=our_summary.get("structure", ""),
            our_implementation=our_summary.get("implementation", ""),
            our_effect=our_summary.get("effect", ""),

            comp_problem=comp_summary.get("problem", ""),
            comp_function=comp_summary.get("function", ""),
            comp_structure=comp_summary.get("structure", ""),
            comp_implementation=comp_summary.get("implementation", ""),
            comp_effect=comp_summary.get("effect", "")
        )

        # LLM 호출 및 결과 파싱
        response = llm.invoke(filled_prompt)

        try:
            result = extract_json_from_llm_output(response)
        except json.JSONDecodeError:
            result = {
                "comparison_axes": [],
                "overall_diff_summary": "JSON parsing failed. Raw output: " + response
            }

        # 비교축별 내용 정리
        rows = []
        for axis_info in result.get("comparison_axes", []):
            rows.append({
                "Comparison Axis": axis_info.get("axis", "-"),
                f"Our ({our_id})": axis_info.get("ours", "-"),
                f"Competitor ({comp_id})": axis_info.get("competitor", "-"),
                "Relevance": axis_info.get("relevance", "-")
            })

        df = pd.DataFrame(rows)

        # 추가 정보는 DataFrame의 속성으로 저장
        df.attrs["overall_diff_summary"] = result.get("overall_diff_summary", "-")
        df.attrs["competitor_id"] = comp_id

        print(df)
        comparison_tables.append(df)

    return comparison_tables
