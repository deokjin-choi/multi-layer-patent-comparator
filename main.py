import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# 상위 디렉토리 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from frontend.components import input_form, progress_tracker, plot_strategy, plot_position, plot_implementation
from app.controllers.engine import run_analysis
from app.controllers.strategy import analyze_strategic_direction
from app.controllers.engine import get_or_fetch_with_summary # 최초 크롤링 되는지 확인용
from app.controllers.fetch_patents import fetch_patent_metadata # 최초 크롤링 되는지 확인용


# 페이지 설정
st.set_page_config(page_title="3-Layer Patent Comparison System", layout="wide")


# -------------------------------
# Introduction and Layer Overview
# -------------------------------
st.subheader("What is the 3-Layer Patent Comparison System?")

st.markdown("""
A structured, LLM-powered system that compares your patent with competitors to identify technical advantages, strategic positioning, and architectural differences through three expert-designed analytical layers.
""")

st.subheader("Why is this system necessary?")

st.markdown("""
Traditional patent analysis is often time-consuming, subjective, and difficult to scale or standardize. This system replaces manual interpretation with LLM-based analysis, offering clear and repeatable comparisons, strategic insights aligned with each patent’s purpose, and automated summarization to support faster and more consistent decision-making. It reduces analyst workload while significantly enhancing the depth and reliability of insights.
""")

st.subheader("Layer Overview")

st.markdown("""

1. **Strategic Direction – Provides a strategic guide for how your patent should be positioned or developed further.** 
   - How it works:  
     - Compares your patent against multiple competitors.  
     - Assesses similarity and relative value.  
     - LLM generates a strategic recommendation based on aggregated insights.

2. **Technology Positioning – Determines which side has the technological and strategic upper hand.**  
   - How it works:  
     - Compares each patent pair across three dimensions:  
       - Functional Purpose  
       - Technical Uniqueness  
       - Strategic Value  
     - LLM determines axis-level winners and provides an overall judgment per comparison.

3. **Implementation Differentiation – Highlights how your technical design differs in structure and approach.**  
   - How it works:  
     - Selects 3–5 technical comparison axes per patent pair.  
     - Summarizes each side’s solution on those axes.  
     - LLM extracts and explains key technical differences.

Each layer supports better decision-making for R&D, patent defense, and strategic investment.
""")

st.subheader("How to Use This System")

st.markdown("""
Before using this system, a prior filtering process has already been completed.  
Your team has:

- Identified key patents from your company and major competitors
- Selected patents that are strategically important within a specific technology area

This system is designed to analyze those patents in a structured, repeatable, and AI-powered way.

### Basic Usage

- Step 1: Enter your target patent number (one patent from your company)
- Step 2: Enter one or more competitor patent numbers
- Step 3: Click the "Run Analysis" button to start the comparison


Once executed, the system will:

- Compare each competitor patent with yours on a one-to-one basis
- Analyze three perspectives using large language models (LLMs):
  - Strategic Direction
  - Technology Positioning
  - Implementation Differentiation
- Provide structured results in the form of tables, summaries, and strategy recommendations

This process supports more informed and objective decision-making in R&D, IP strategy, and technology planning.
""")


# -------------------------------
# Step 1: User Input
# -------------------------------
user_input = input_form.render()
result_path = "data/final_results/last_result.pkl"

# streamlit 내에서 정의
status_text = st.empty()
progress_bar = st.progress(0)

# -------------------------------
# Step 2: Run New Analysis
# -------------------------------
if user_input["submitted"]:
    if not user_input["our_patent"].strip():
        st.warning("Please enter your patent number before running the analysis.")
        st.stop()
    elif not user_input["competitor_patents"]:
        st.warning("Please enter at least one competitor patent number.")
        st.stop()
    else:
        # Step 1️⃣: 당사 특허 확인
        our_patent_check = fetch_patent_metadata(user_input["our_patent"])
        if our_patent_check is None:
            st.error("❌ Our company patent not found. Please check the patent number.")
            st.stop()

        # Step 2️⃣: 경쟁사 특허 유효성 필터링
        valid_competitors = []
        invalid_competitors = []
        for pid in user_input["competitor_patents"]:
            result = fetch_patent_metadata(pid)
            if result:
                valid_competitors.append(pid)
            else:
                invalid_competitors.append(pid)

        # 당사 특허가 경쟁사 목록에 포함된 경우 제거
        our_patent_id = user_input["our_patent"].strip()        
        if our_patent_id in valid_competitors:
            valid_competitors.remove(our_patent_id)
            st.warning(f"⚠️ Our patent ID '{our_patent_id}' was also entered as a competitor. It has been excluded.")


        # 경쟁사 전부 실패한 경우
        if not valid_competitors:
            st.error("❌ None of the competitor patents could be analyzed. Please check the patent numbers.")
            st.stop()

        # 일부 실패한 경우 경고 표시
        if invalid_competitors:
            st.warning(
                f"⚠️ {len(invalid_competitors)} competitor patent(s) could not be analyzed and will be excluded: {', '.join(invalid_competitors)}"
            )

        pos_result, imp_diff_result, imp_diff_by_axis = run_analysis(
            our_patent_id=user_input["our_patent"],
            competitor_patent_ids=valid_competitors, # 유효한 경쟁사 특허만 전달
            status_text=status_text, 
            progress_bar=progress_bar, 
            show_progress=progress_tracker.show_progress
        )

        progress_tracker.show_progress("strategy", status_text, progress_bar)
        strategy_output = analyze_strategic_direction(pos_result, user_input["our_patent"])

        with open(result_path, "wb") as f:
            pickle.dump({
                "pos_result": pos_result,
                "imp_diff_result": imp_diff_result,
                "imp_diff_by_axis": imp_diff_by_axis,
                "strategy_output": strategy_output
            }, f)

        st.success("Analysis completed successfully. You can download the report from the sidebar.")

# -------------------------------
# Step 3: Load Cached Results
# -------------------------------
elif os.path.exists(result_path):
    with open(result_path, "rb") as f:
        cached_result = pickle.load(f)
        pos_result = cached_result["pos_result"]
        imp_diff_result = cached_result["imp_diff_result"]
        imp_diff_by_axis = cached_result["imp_diff_by_axis"]
        strategy_output = cached_result["strategy_output"]
    st.info("Previous analysis loaded. See below for the results.")
else:
    st.info("No analysis result available yet.")
    pos_result = imp_diff_result = imp_diff_by_axis = strategy_output = None

# -------------------------------
# Step 4: Display Results
# -------------------------------

if pos_result and imp_diff_result and imp_diff_by_axis and strategy_output:
    # Strategic Direction
    st.subheader("(1) Strategic Direction Result")
    
    st.markdown("#### Strategic Recommendation")
    st.success(strategy_output["our_overall_strategy"])

    print("Strategic Positioning Scores:", strategy_output["strategy_scores"])

    if "strategy_scores" in strategy_output:
        col1, col2 = st.columns([1, 1])

        # 🎯 왼쪽: 레이더 차트
        with col1:
            st.markdown("##### Strategic Positioning Overview")
            plot_strategy.plot_strategy_radar(strategy_output["strategy_scores"])

        # 🧠 오른쪽: 핵심 점수 + 이유 요약 테이블
    with col2:
        st.markdown("##### Key Strategic Scores by Aspect")

        scores = strategy_output["strategy_scores"]

        # 점수/이유 분리 → 각 열 구성
        records = []
        for item in scores:
            pid = item["patent_id"]
            for aspect, val in item["scores"].items():
                records.append({
                    "Aspect": aspect,
                    f"{pid} - Score": val["score"],
                    f"{pid} - Reason": val["reason"]
                })

        df_combined = pd.DataFrame(records)

        # ✅ 커스텀 순서로 정렬
        custom_order = [
            "Performance Advantage", "Core Tech Area", "Competitor Block",
            "Market Impact", "Application Versatility", "Differentiation"
        ]
        df_combined["Aspect"] = pd.Categorical(df_combined["Aspect"], categories=custom_order, ordered=True)
        df_sorted = df_combined.sort_values("Aspect")

        # ✅ 병합 정리
        df_grouped = df_sorted.groupby("Aspect", as_index=False, observed=True).first()

        # ✅ 폰트 크기 확대 및 행간 확보용 스타일 삽입 (HTML 스타일)
        st.markdown(
            """
            <style>
            .element-container table td {
                font-size: 16px !important;
                padding-top: 12px !important;
                padding-bottom: 12px !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # ✅ 높이와 정렬 반영한 출력
        st.dataframe(df_grouped, use_container_width=True, hide_index=True, height=500)

        

    st.markdown("#### Patent Summary Table")
    df_strategy = pd.DataFrame(strategy_output["strategy_table"])
    st.table(df_strategy[["patent_id","assignee", "tech_summary", "technical_value", "strategic_direction"]].reset_index(drop=True))

    # Technology Positioning
    st.subheader("(2) Technology Positioning Result")
    plot_position.render_positioning_summary_table(pos_result)


    # 실제 1:1 비교 결과 테이블
    for df in pos_result:
        st.markdown(f"#### Our Patent vs Competitor ({df.attrs.get('competitor_id', 'Unknown')})")
        st.table(df.reset_index(drop=True))
        st.markdown(f"**Overall Winner:** `{df.attrs.get('overall_winner', 'N/A')}`")
        st.markdown(f"**Reason:** {df.attrs.get('overall_judgement', 'N/A')}")
        st.markdown("---")

    # Implementation Differentiation
    st.subheader("(3) Implementation Differentiation Result")

    # 당사 특허 id 설정(캐시에 있을 떄도)
    our_id = (
    user_input["our_patent"]
    if user_input.get("our_patent") and user_input["submitted"]
    else strategy_output["strategy_table"][0]["patent_id"]
    if strategy_output and "strategy_table" in strategy_output
    else "UNKNOWN"
    )

    plot_implementation.render_implementation_tree(axis_summary = imp_diff_by_axis['axis_summary'], our_id=our_id)
    
    for df in imp_diff_result:
        st.markdown(f"#### Our Patent vs Competitor ({df.attrs.get('competitor_id', 'Unknown')})")
        st.table(df.reset_index(drop=True))
        st.markdown(f"**Summary of Technical Difference:** {df.attrs.get('overall_diff_summary', 'N/A')}")
        st.markdown("---")

    if st.button("Reset Analysis Results"):
        os.remove(result_path)
        st.experimental_rerun()
