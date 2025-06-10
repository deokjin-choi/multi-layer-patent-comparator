from app.utils.prompts import load_prompt
from app.utils.json_helper import extract_json_from_llm_output  # 위 함수 별도 분리 시
from app.utils.llm.llm_factory import get_llm_client
from app.utils.path_helper import RAW_DATA_DIR
from app.utils.llm.retry_utils import safe_invoke


import pandas as pd

import os
import json

# 1️⃣ 데이터 폴더 경로 지정 (예: data/raw)
import os

def analyze_strategic_direction(pos_result: list[pd.DataFrame], our_patent_id: str):

    # 1. Extract our patent summary from the first df
    first_df = pos_result[0]
    our_cols = [col for col in first_df.columns if col.startswith("Our Summary")]
    our_fp = first_df[first_df["Aspect"] == "Functional Purpose"][our_cols[0]].values[0]
    our_tu = first_df[first_df["Aspect"] == "Technical Uniqueness"][our_cols[0]].values[0]
    our_sv = first_df[first_df["Aspect"] == "Strategic Value"][our_cols[0]].values[0]

    # our assignee 정보 추가
    our_raw_file_path = os.path.join(RAW_DATA_DIR, our_patent_id+ ".json")

    with open(our_raw_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        our_assignee = data.get("assignee", None)

    competitors = []
    high_value_count = 0
    low_value_count = 0
    medium_value_count = 0

    for df in pos_result:
        comp_id = df.attrs["competitor_id"]
        overall = df.attrs["overall_winner"]
        reason = df.attrs["overall_judgement"]

        # comp assignee 정보 추가
        comp_raw_file_path = os.path.join(RAW_DATA_DIR, comp_id+ ".json")

        with open(comp_raw_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            comp_assignee = data.get("assignee", None)

        comp_cols = [col for col in df.columns if f"({comp_id})" in col]
        fp = df[df["Aspect"] == "Functional Purpose"][comp_cols[0]].values[0]
        tu = df[df["Aspect"] == "Technical Uniqueness"][comp_cols[0]].values[0]
        sv = df[df["Aspect"] == "Strategic Value"][comp_cols[0]].values[0]

        if overall == "competitor":
            tech_value = "High"
            high_value_count += 1
        elif overall == "ours":
            tech_value = "Low"
            low_value_count += 1
        elif overall == "tie":
            tech_value = "Medium"
            medium_value_count += 1

        competitors.append({
            "id": comp_id,
            "assignee": comp_assignee,  # 경쟁사 특허의 assignee 정보 추가
            "fp": fp,
            "tu": tu,
            "sv": sv,
            "evaluation": overall,
            "reason": reason,
            "similarity": None,
            "value": tech_value
        })

    total = len(competitors)

    high_ratio = high_value_count / total
    low_ratio = low_value_count / total

    # Determine our value based on high/low ratios
    our_value = "Medium"
    if high_ratio >= 0.5:
        our_value = "Low"
    elif low_ratio >= 0.5:
        our_value = "High"

    # competitor YAML block generation
    competitor_str = ""
    for c in competitors:
        competitor_str += f"""- ID: {c["id"]}
  Assignee: {c["assignee"]}
  FP: {c["fp"]}
  TU: {c["tu"]}
  SV: {c["sv"]}
  Evaluation: {c["evaluation"]} (Reason: {c["reason"]})
  TechnicalValue: {c["value"]}
"""

    # Load template
    template = load_prompt("strategy", "direction_v4")

    # Fill in final prompt
    prompt = template.format(
        our_id=our_patent_id,
        our_assignee=our_assignee,
        our_fp=our_fp,
        our_tu=our_tu,
        our_sv=our_sv,
        our_value=our_value,
        competitors_section=competitor_str
    )

    llm = get_llm_client()

    # Call LLM
    response = safe_invoke(llm, prompt, extract_json_from_llm_output)

    if response is None:
        response = {
            "strategy_table": [],
            "our_overall_strategy": "LLM failed after retries or JSON parsing failed."
        }

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

    # 3. 새로운 포지셔닝 점수 프롬프트 추가 실행 - 프롬프트 채워넣기
    score_template = load_prompt("strategy", "position_score_v2")
    score_prompt = score_template.format(patents_table=score_input_yaml)

    # 4. LLM 호출 및 결과 파싱
    score_data = safe_invoke(llm, score_prompt, extract_json_from_llm_output)

    if score_data is None:
        score_data = []

    return {
    "our_overall_strategy": response["our_overall_strategy"],
    "strategy_table": response["strategy_table"],
    "strategy_scores": score_data  # ← Radar 등에서 활용 가능
}