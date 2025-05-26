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

# 추상화된 관점에서 차이점 요약 -> 1:N 전체 시각화를 위해서
def analyze_implementation_diff_by_axis(imp_diff_result: list[pd.DataFrame], our_id: str) -> dict:
    llm = get_llm_client()
    prompt_template = load_prompt("implementation", "axis_standard_v4")  # 프롬프트 헤더

    # 누적 입력용 리스트
    merged_input = []

    for df in imp_diff_result:
        comp_id = df.attrs.get("competitor_id", "unknown")

        axes = []
        for _, row in df.iterrows():
            axis_name = row["Comparison Axis"]
            ours_desc = row.get(f"Our ({our_id})", "not explicitly described")
            comp_desc = row.get(f"Competitor ({comp_id})", "not explicitly described")
            axes.append({
                "axis": axis_name,
                "ours": ours_desc,
                "competitor": comp_desc
            })

        merged_input.append({
            "our_patent_id": our_id,
            "competitor_patent_id": comp_id,
            "axes": axes
        })

    # 전체 입력을 단일 JSON에 넣기
    input_json = json.dumps({
        "input_list": merged_input  # ← axis_standard_v4 프롬프트에서 input_list 기준
    }, ensure_ascii=False, indent=2)

    # 전체 프롬프트 구성 및 출력
    full_prompt = prompt_template + "\n\nInput Data:\n" + input_json
    print("[DEBUG] Full Prompt:\n", full_prompt)

    # LLM 호출 한 번만 수행
    llm_response = llm.invoke(full_prompt)

    try:
        parsed = extract_json_from_llm_output(llm_response)
        return {
            "axis_summary": parsed.get("implementation_summary", [])
        }
    except Exception as e:
        print(f"[ERROR] JSON parsing failed: {e}")
        return {
            "axis_summary": [],
            "error": str(e)
        }

