import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd


# positoning 결과를 요약하는 함수
def summarize_positioning_advantage(pos_result):
    summary_count = {
        "functional_purpose": {"ours": 0, "competitor": 0, "tie": 0},
        "technical_uniqueness": {"ours": 0, "competitor": 0, "tie": 0},
        "strategic_value": {"ours": 0, "competitor": 0, "tie": 0}
    }

    for df in pos_result:
        for aspect in ["functional_purpose", "technical_uniqueness", "strategic_value"]:
            winner = df.loc[df["Aspect"].str.lower() == aspect.replace("_", " "), "Winner"].values
            if len(winner) == 0:
                continue
            winner_value = winner[0].strip().lower()
            if winner_value in summary_count[aspect]:
                summary_count[aspect][winner_value] += 1

    # 카운트 추출
    fp = summary_count["functional_purpose"]
    tu = summary_count["technical_uniqueness"]
    sv = summary_count["strategic_value"]

    total = sum(fp.values())

    def dominant(counts):
        if counts["ours"] > counts["competitor"] and counts["ours"] >= counts["tie"]:
            return "ours"
        elif counts["competitor"] > counts["ours"] and counts["competitor"] >= counts["tie"]:
            return "competitor"
        else:
            return "tie"

    # 각 Layer별 해석
    fp_dominant = dominant(fp)
    tu_dominant = dominant(tu)
    sv_dominant = dominant(sv)

    # Functional Purpose (핵심 문제성 + 산업적 영향력 + 규제 시급성)
    fp_text = {
        "ours": f"Our patent leads in functional purpose ({fp['ours']} out of {total}), indicating stronger focus on solving critical industry problems with broad applicability.",
        "competitor": f"Competitor patents dominate in functional purpose ({fp['competitor']} out of {total}), suggesting their alignment with higher-impact challenges or more urgent regulatory demands.",
        "tie": "Functional purpose results were balanced, showing both sides address similarly important and broadly relevant problems."
    }[fp_dominant]

    # Technical Uniqueness (구조적 참신성 + 기술적 깊이 + 구현 복잡도)
    tu_text = {
        "ours": f"We demonstrated higher technical uniqueness ({tu['ours']} out of {total}), based on structurally innovative mechanisms and technically deep implementation approaches.",
        "competitor": f"Competitors showed greater technical uniqueness ({tu['competitor']} out of {total}), reflecting novel structural ideas and more complex implementation strategies.",
        "tie": "Technical uniqueness was evenly split, indicating comparable innovation in structure and execution between both sides."
    }[tu_dominant]

    # Strategic Value (비용 효율성 + 규제 정합성 + 시장성)
    sv_text = {
        "ours": f"Our patent delivered stronger strategic value ({sv['ours']} out of {total}), with advantages in cost efficiency, quality improvement, and commercial potential.",
        "competitor": f"Competitor patents showed higher strategic value ({sv['competitor']} out of {total}), suggesting stronger market potential and alignment with business needs.",
        "tie": "Strategic value results were evenly matched, indicating similar levels of cost, compliance, and market potential."
    }[sv_dominant]


    return f"{fp_text} {tu_text} {sv_text}"


def render_positioning_summary_table(pos_result):
    """
    Render a summary table comparing positioning results (FP, TU, SV) between our patent and competitors.
    Uses emoji colors to indicate which side was favored in each aspect.
    """
    if not pos_result:
        st.warning("No positioning result to summarize.")
        return

    heatmap_data = []
    headers = ["Competitor", "Functional Purpose", "Technical Uniqueness", "Strategic Value"]

    for df in pos_result:
        row = [df.attrs.get("competitor_id", "Unknown")]
        for aspect in ["functional_purpose", "technical_uniqueness", "strategic_value"]:
            winner_series = df.loc[df["Aspect"].str.lower() == aspect.replace("_", " "), "Winner"]
            if len(winner_series) == 0:
                row.append("N/A")
            else:
                winner = winner_series.values[0].strip().lower()
                if winner == "ours":
                    row.append("🟩 ours")
                elif winner == "competitor":
                    row.append("🟥 comp")
                elif winner == "tie":
                    row.append("🟨 tie")
                else:
                    row.append("N/A")
        heatmap_data.append(row)

    df_heatmap = pd.DataFrame(heatmap_data, columns=headers)
    txt_summary = summarize_positioning_advantage(pos_result)
    st.markdown("#### Summary of Positioning Comparison Table")
    st.markdown(txt_summary)
    st.table(df_heatmap.reset_index(drop=True))