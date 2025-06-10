# controller/positioning.py

from app.utils.llm.llm_factory import get_llm_client
from app.utils.json_helper import extract_json_from_llm_output  # 위 함수 별도 분리 시
from app.utils.prompts import load_prompt
import logging
import pandas as pd
from app.utils.llm.retry_utils import safe_invoke


logger = logging.getLogger(__name__)

def generate_positioning_summary(patent_id: str, summary: dict, prompt_version="summarize_v1"):
    """problem~effect로부터 FP, TU, SV 생성"""
    try:
        prompt_template = load_prompt("positioning", prompt_version)
        llm_input = {
            "problem": summary["problem"],
            "function": summary["solution_function"],
            "structure": summary["solution_structure"],
            "implementation": summary["solution_implementation"],
            "effect": summary["effect"]
        }
        prompt = prompt_template.format(**llm_input)

        llm = get_llm_client()
        response = safe_invoke(llm, prompt, extract_json_from_llm_output)

        if response is None:
            response = {
                "functional_purpose": "N/A",
                "technical_uniqueness": "N/A",
                "strategic_value": "N/A"
            }

        return response

def compare_positioning(our_id: str, our_pos: dict, comp_id: str, comp_pos: dict, prompt_version="diff_v1"):
    """당사 vs 경쟁사 1:1 비교"""
    try:
        # diff_v7: 양방향 평가 + confidence 기반 선택
        if prompt_version == "diff_v7" or prompt_version == "diff_v8":
            prompt_template = load_prompt("positioning", prompt_version)

            # A-first prompt (ours first)
            prompt_a_first = prompt_template.format(
                patent_id_a=our_id,
                patent_a_fp=our_pos["functional_purpose"],
                patent_a_tu=our_pos["technical_uniqueness"],
                patent_a_sv=our_pos["strategic_value"],
                patent_id_b=comp_id,
                patent_b_fp=comp_pos["functional_purpose"],
                patent_b_tu=comp_pos["technical_uniqueness"],
                patent_b_sv=comp_pos["strategic_value"]
            )

            # B-first prompt (competitor first)
            prompt_b_first = prompt_template.format(
                patent_id_a=comp_id,
                patent_a_fp=comp_pos["functional_purpose"],
                patent_a_tu=comp_pos["technical_uniqueness"],
                patent_a_sv=comp_pos["strategic_value"],
                patent_id_b=our_id,
                patent_b_fp=our_pos["functional_purpose"],
                patent_b_tu=our_pos["technical_uniqueness"],
                patent_b_sv=our_pos["strategic_value"]
            )

            llm = get_llm_client()
            
            result_a = safe_invoke(llm, prompt_a_first, extract_json_from_llm_output)
            result_b = safe_invoke(llm, prompt_b_first, extract_json_from_llm_output)

            if result_a is None:
                result_a = {
                    "aspect_evaluation": {},
                    "overall_winner": "competitor",
                    "overall_judgement": "A-first comparison failed; defaulting to competitor.",
                    "confidence": 0.0
                }
            if result_b is None:
                result_b = {
                    "aspect_evaluation": {},
                    "overall_winner": "ours",
                    "overall_judgement": "B-first comparison failed; defaulting to ours.",
                    "confidence": 0.0
                }        

            # confidence 비교
            final_result = result_a if result_a.get("confidence", 0) >= result_b.get("confidence", 0) else result_b

            # patent_id → ours / competitor 변환
            def map_role(pid):
                if pid == our_id:
                    return "ours"
                elif pid == comp_id:
                    return "competitor"
                elif pid == "tie":
                    return "tie"
                else:
                    return pid  # 예외

            remapped = {
                "aspect_evaluation": {
                    k: {
                        "winner": map_role(v["winner"]),
                        "reason": v["reason"]
                    } for k, v in final_result["aspect_evaluation"].items()
                },
                "overall_winner": map_role(final_result["overall_winner"]),
                "overall_judgement": final_result["overall_judgement"]
            }

            return remapped

        # diff_v6: 기존 방식 (A = ours, B = competitor)
        elif prompt_version == "diff_v6":
            prompt_template = load_prompt("positioning", prompt_version)
            prompt = prompt_template.format(
                patent_a_fp=our_pos["functional_purpose"],
                patent_a_tu=our_pos["technical_uniqueness"],
                patent_a_sv=our_pos["strategic_value"],
                patent_b_fp=comp_pos["functional_purpose"],
                patent_b_tu=comp_pos["technical_uniqueness"],
                patent_b_sv=comp_pos["strategic_value"]
            )
            llm = get_llm_client()
            response = llm.invoke(prompt)
            raw_result = extract_json_from_llm_output(response)

            def remap_winner(value):
                return "ours" if value == "A" else "competitor"

            def remap_reason(text):
                return text.replace("Patent A", "Our patent").replace("Patent B", "Competitor patent")

            remapped = {
                "aspect_evaluation": {
                    k: {
                        "winner": remap_winner(v["winner"]),
                        "reason": remap_reason(v["reason"])
                    } for k, v in raw_result["aspect_evaluation"].items()
                },
                "overall_winner": remap_winner(raw_result["overall_winner"]),
                "overall_judgement": remap_reason(raw_result["overall_judgement"])
            }
            return remapped

        # 그 외 버전은 기존 방식
        else:
            prompt_template = load_prompt("positioning", prompt_version)
            prompt = prompt_template.format(
                our_fp=our_pos["functional_purpose"],
                our_tu=our_pos["technical_uniqueness"],
                our_sv=our_pos["strategic_value"],
                comp_fp=comp_pos["functional_purpose"],
                comp_tu=comp_pos["technical_uniqueness"],
                comp_sv=comp_pos["strategic_value"]
            )
            llm = get_llm_client()
            response = llm.invoke(prompt)
            return extract_json_from_llm_output(response)

    except Exception as e:
        logger.error(f"[{our_id} vs {comp_id}] Positioning comparison failed: {e}")
        return {
            "aspect_evaluation": {},
            "overall_winner": "competitor",
            "overall_judgement": "Comparison failed; defaulting to competitor.",
            "confidence": 0.0
        }

 

def analyze_positioning(our_patent: dict, competitor_patents: list[dict]) -> list[pd.DataFrame]:
    """당사 특허 vs 각 경쟁사 특허에 대한 비교 결과를 DataFrame 리스트로 리턴"""

    our_id = our_patent["id"]
    our_summary = {k: v for k, v in our_patent.items() if k != "id"}
    our_pos = generate_positioning_summary(our_id, our_summary, prompt_version="summarize_v3")
    print(f"Positioning summary for {our_id}: {our_pos}")
    comparison_tables = []

    for comp in competitor_patents:
        comp_id = comp["id"]
        comp_summary = {k: v for k, v in comp.items() if k != "id"}
        comp_pos = generate_positioning_summary(comp_id, comp_summary, prompt_version="summarize_v3")

        result = compare_positioning(our_id, our_pos, comp_id, comp_pos, prompt_version="diff_v8")
        print(f"Comparison result for {our_id} vs {comp_id}: {result}")

        # 테이블 형태로 변환
        aspects = ["functional_purpose", "technical_uniqueness", "strategic_value"]
        aspect_labels = {
            "functional_purpose": "Functional Purpose",
            "technical_uniqueness": "Technical Uniqueness",
            "strategic_value": "Strategic Value"
        }

        rows = []
        for aspect in aspects:
            rows.append({
                "Aspect": aspect_labels[aspect],
                f"Our Summary ({our_id})": our_pos[aspect],
                f"Competitor Summary ({comp_id})": comp_pos[aspect],
                "Winner": result["aspect_evaluation"].get(aspect, {}).get("winner", "-"),
                "Reason": result["aspect_evaluation"].get(aspect, {}).get("reason", "-")
            })

        df = pd.DataFrame(rows)
        
        df.attrs["overall_winner"] = result.get("overall_winner", "-")
        df.attrs["overall_judgement"] = result.get("overall_judgement", "-")
        df.attrs["competitor_id"] = comp_id
        

        print(df)

        comparison_tables.append(df)

    return comparison_tables
