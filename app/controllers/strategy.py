from app.utils.prompts import load_prompt
from app.utils.json_helper import extract_json_from_llm_output  # 위 함수 별도 분리 시
from app.utils.llm.llm_factory import get_llm_client

import pandas as pd

def analyze_strategic_direction(pos_result: list[pd.DataFrame], our_patent_id: str):

    # 1. Extract our patent summary from the first df
    first_df = pos_result[0]
    our_cols = [col for col in first_df.columns if col.startswith("Our Summary")]
    our_fp = first_df[first_df["Aspect"] == "Functional Purpose"][our_cols[0]].values[0]
    our_tu = first_df[first_df["Aspect"] == "Technical Uniqueness"][our_cols[0]].values[0]
    our_sv = first_df[first_df["Aspect"] == "Strategic Value"][our_cols[0]].values[0]

    competitors = []
    high_value_count = 0
    low_value_count = 0

    for df in pos_result:
        comp_id = df.attrs["competitor_id"]
        overall = df.attrs["overall_winner"]
        reason = df.attrs["overall_judgement"]

        comp_cols = [col for col in df.columns if f"({comp_id})" in col]
        fp = df[df["Aspect"] == "Functional Purpose"][comp_cols[0]].values[0]
        tu = df[df["Aspect"] == "Technical Uniqueness"][comp_cols[0]].values[0]
        sv = df[df["Aspect"] == "Strategic Value"][comp_cols[0]].values[0]

        tech_value = "High" if overall == "competitor" else "Low"
        if tech_value == "High":
            high_value_count += 1
        elif tech_value == "Low":
            low_value_count += 1

        competitors.append({
            "id": comp_id,
            "fp": fp,
            "tu": tu,
            "sv": sv,
            "evaluation": overall,
            "reason": reason,
            "similarity": None,# 나중에 다시 계산됨
            "value": tech_value
        })

    total = len(competitors)
    our_value = "Medium"
    if high_value_count / total >= 0.6:
        our_value = "Low"
    elif low_value_count / total >= 0.6:
        our_value = "High"

    # competitor YAML block generation
    competitor_str = ""
    for c in competitors:
        competitor_str += f"""- ID: {c["id"]}
  FP: {c["fp"]}
  TU: {c["tu"]}
  SV: {c["sv"]}
  Evaluation: {c["evaluation"]} (Reason: {c["reason"]})
  TechnicalValue: {c["value"]}
"""

    # Load template
    template = load_prompt("strategy", "direction_v2")

    # Fill in final prompt
    prompt = template.format(
        our_id=our_patent_id,
        our_fp=our_fp,
        our_tu=our_tu,
        our_sv=our_sv,
        our_value=our_value,
        competitors_section=competitor_str
    )

    llm = get_llm_client()

    # Call LLM
    result = llm.invoke(prompt)
    print(prompt)
    response = extract_json_from_llm_output(result)  # JSON 파싱

   
    # 👇 새로운 포지셔닝 점수 프롬프트 추가 실행
    score_template = load_prompt("strategy", "position_score_v1")

    # 1. our patent 정보 구성
    strategy_table = response["strategy_table"]  # ← 기존 프롬프트에서 생성된 결과 그대로 활용
    score_input_list = []

    for entry in strategy_table:
        score_input_list.append({
            "patent_id": entry["patent_id"],
            "tech_summary": entry["tech_summary"],
            "strategic_direction": entry["strategic_direction"],
            "technical_value": entry["technical_value"]
        })

    # 2. YAML-like 문자열로 변환 (프롬프트 안에 삽입할 형태로)    
    score_input_yaml = ""
    for item in strategy_table:
        score_input_yaml += f"""- patent_id: {item["patent_id"]}
    tech_summary: {item["tech_summary"]}
    strategic_direction: {item["strategic_direction"]}
    technical_value: {item["technical_value"]}
    """

    # 3. 프롬프트 채워넣기
    score_prompt = score_template.format(patents_table=score_input_yaml)

    # 4. LLM 호출 및 결과 파싱
    score_result = llm.invoke(score_prompt)
    score_data = extract_json_from_llm_output(score_result)  # JSON list 형태 반환

    return {
    "our_overall_strategy": response["our_overall_strategy"],
    "strategy_table": response["strategy_table"],
    "strategy_scores": score_data  # ← Radar 등에서 활용 가능
}