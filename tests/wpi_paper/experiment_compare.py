# %%
import sys
import os
import json
import pandas as pd

import glob


from tqdm import tqdm

# app 폴더가 있는 프로젝트 루트를 path에 추가
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))

import importlib
import tests.wpi_paper.metrics_eval as metrics_eval

importlib.reload(metrics_eval)


from app.controllers.engine import get_or_fetch_with_summary
from app.controllers.positioning import analyze_positioning

from tests.wpi_paper.metrics_eval import evaluate_comparison_results, compare_results_overall_table, compare_detailed_results_by_domain, compare_shakiness_summary

# %%
# 기술 도메인과 특허 매핑 정보
domain_patents = {
    "AI": {
        "ours": ["US11475102B2"],
        "competitors": ["US20230297372A1", "CN109992743B", "US11347652B2", "CN109478252B"]
    },
    "Energy": {
        "ours": ["US9257711B2"],
        "competitors": ["US11955674B1", "EP4469183A1", "US9502728B1", "US10252243B2"]
    },
    "Consumer": {
        "ours": ["US10231596B2"],
        "competitors": ["US11219349B2", "US12029369B2", "EP3560405A1","KR102528669B1"]
    },
    "Mobile": {
        "ours": ["US11703916B2"],
        "competitors": ["CN114909388B", "EP4557048A1", "US20240430348A1", "JP2022165707A"]
    },
    "Healthcare": {
        "ours": ["US20210259560A1"],
        "competitors": ["KR102407094B1", "US11432766B2", "US9348322B2", "US9775548B2"]
    }
}

# %%

# 버전에 따라 저장 폴더 지정
prompt_version = "diff_v9"  # 또는 "diff_v8"
save_dir = "old" if prompt_version == "diff_v1" else "new"
os.makedirs(save_dir, exist_ok=True)

# 30회 반복
for trial in tqdm(range(1, 31), desc="Running 30 Trials"):
    save_path = f"{save_dir}/trial_{trial}.csv"
    if os.path.exists(save_path):
        continue  # 이미 존재하면 스킵

    results = []

    for domain, ids in domain_patents.items():
        our_id = ids["ours"][0]
        comp_ids = ids["competitors"]

        our_patent = get_or_fetch_with_summary(our_id)
        competitor_patents = [get_or_fetch_with_summary(pid) for pid in comp_ids]

        # 기존 analyze_positioning 호출
        pos_result_list = analyze_positioning(
            our_patent["summary"],
            [cp["summary"] for cp in competitor_patents]
        )

        for comp_id, df in zip(comp_ids, pos_result_list):
            aspect_rows = df.to_dict(orient="records")

            aspect_map = {
                row["Aspect"]: {
                    "winner": row["Winner"],
                    "reason": row["Reason"],
                    "our_summary": row.get(f"Our Summary ({our_id})", "-"),
                    "comp_summary": row.get(f"Competitor Summary ({comp_id})", "-")
                } for row in aspect_rows
            }

            results.append({
                "domain": domain,
                "trial": trial,
                "our_id": our_id,
                "competitor_id": comp_id,
                "fp_winner": aspect_map.get("Functional Purpose", {}).get("winner", "-"),
                "fp_reason": aspect_map.get("Functional Purpose", {}).get("reason", "-"),
                "fp_ours": aspect_map.get("Functional Purpose", {}).get("our_summary", "-"),
                "fp_competitor": aspect_map.get("Functional Purpose", {}).get("comp_summary", "-"),
                "tu_winner": aspect_map.get("Technical Uniqueness", {}).get("winner", "-"),
                "tu_reason": aspect_map.get("Technical Uniqueness", {}).get("reason", "-"),
                "tu_ours": aspect_map.get("Technical Uniqueness", {}).get("our_summary", "-"),
                "tu_competitor": aspect_map.get("Technical Uniqueness", {}).get("comp_summary", "-"),
                "sv_winner": aspect_map.get("Strategic Value", {}).get("winner", "-"),
                "sv_reason": aspect_map.get("Strategic Value", {}).get("reason", "-"),
                "sv_ours": aspect_map.get("Strategic Value", {}).get("our_summary", "-"),
                "sv_competitor": aspect_map.get("Strategic Value", {}).get("comp_summary", "-"),
                "overall_winner": df.attrs.get("overall_winner", "-"),
                "overall_reason": df.attrs.get("overall_judgement", "-")
            })

    # trial 별 저장
    df_trial = pd.DataFrame(results)
    df_trial.to_csv(save_path, index=False)

# 전체 결과 통합 저장
all_csvs = glob.glob(f"{save_dir}/trial_*.csv")
df_full = pd.concat([pd.read_csv(f) for f in all_csvs], ignore_index=True)
df_full.to_csv(f"{save_dir}/trial_full.csv", index=False)

# %% 비교실시
new_df = pd.read_csv("new/trial_full.csv")
old_df = pd.read_csv("old/trial_full.csv")

old_result, new_result = evaluate_comparison_results(old_df, new_df)

# %%

comparison_table = compare_results_overall_table(old_result, new_result)
print(comparison_table)

summary = compare_shakiness_summary(old_df, new_df)

print(summary["overall_comparison"])
print(summary["domain_comparison"])

# %%
comparison_table_domain = compare_detailed_results_by_domain(old_result, new_result)
print(comparison_table_domain)