# %%
import sys
import os
import json
import pandas as pd

import glob
import matplotlib.pyplot as plt


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
# === STEP 1: 각 winner type의 분포 계산 함수 ===
def get_winner_distribution(df, column):
    """
    특정 winner column (e.g., 'fp_winner')에서 ours / competitor / tie 비율 계산
    """
    dist = df[column].value_counts(normalize=True) * 100
    return {
        'ours': dist.get('ours', 0),
        'competitor': dist.get('competitor', 0),
        'tie': dist.get('tie', 0)
    }

# === STEP 2: FP, TU, SV 각각에 대해 승자 분포 계산 ===
def prepare_all_distributions(old_df, new_df):
    return {
        'FP': (get_winner_distribution(old_df, 'fp_winner'),
               get_winner_distribution(new_df, 'fp_winner')),
        'TU': (get_winner_distribution(old_df, 'tu_winner'),
               get_winner_distribution(new_df, 'tu_winner')),
        'SV': (get_winner_distribution(old_df, 'sv_winner'),
               get_winner_distribution(new_df, 'sv_winner')),
    }

# === STEP 3: 시각화 함수 ===
def plot_winner_distributions(distributions):
    """
    FP, TU, SV에 대한 승자 분포를 막대 그래프로 시각화 (범례 및 퍼센트 추가)
    """
    labels = ['Original Prompt', 'Revised Prompt']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (dimension, (old_dist, new_dist)) in enumerate(distributions.items()):
        ours = [old_dist['ours'], new_dist['ours']]
        comp = [old_dist['competitor'], new_dist['competitor']]
        tie = [old_dist['tie'], new_dist['tie']]

        bottoms1 = ours
        bottoms2 = [i + j for i, j in zip(ours, comp)]

        ax = axes[idx]
        bars1 = ax.bar(labels, ours, label='Ours', color='skyblue')
        bars2 = ax.bar(labels, comp, bottom=bottoms1, label='Competitor', color='salmon')
        bars3 = ax.bar(labels, tie, bottom=bottoms2, label='Tie', color='lightgreen')

        # 퍼센트 수치 표시 (ours)
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height / 2, f'{height:.1f}%', ha='center', va='center', fontsize=9)

        # 퍼센트 수치 표시 (competitor)
        for bar, base in zip(bars2, bottoms1):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, base + height / 2, f'{height:.1f}%', ha='center', va='center', fontsize=9)

        # 퍼센트 수치 표시 (tie)
        for bar, base in zip(bars3, bottoms2):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, base + height / 2, f'{height:.1f}%', ha='center', va='center', fontsize=9)

        ax.set_ylim(0, 100)
        ax.set_ylabel('Percentage (%)')
        ax.set_title(f'{dimension} Winner Distribution (Percentage)')
        ax.legend(loc='lower left', fontsize=8)

    plt.tight_layout()
    plt.show()


# === STEP 4: 실행 예시 ===
# old_df와 new_df는 사전에 로드되어 있다고 가정
# 예: old_df = pd.read_csv('old.csv'), new_df = pd.read_csv('new.csv')
distributions = prepare_all_distributions(old_df, new_df)
plot_winner_distributions(distributions)



# %%
comparison_table_domain = compare_detailed_results_by_domain(old_result, new_result)

# %%
# ===== Ours Ratio (%) 추출 및 Global Mean 계산 =====
ours_ratio_df = comparison_table_domain[
    comparison_table_domain.index.str.contains("Ours Ratio")
].copy()

ours_ratio_df["Domain"] = ours_ratio_df.index.str.split(" - ").str[0]
ours_ratio_df = ours_ratio_df.rename(columns={
    "Old Prompt": "Original",
    "New Prompt": "Revised"
})[["Domain", "Original", "Revised"]]

# Global Mean 자동 계산 (마지막 행 추가)
global_mean_ours = ours_ratio_df[["Original", "Revised"]].mean().to_frame().T
global_mean_ours.insert(0, "Domain", "Global Mean")
ours_ratio_df = pd.concat([ours_ratio_df, global_mean_ours], ignore_index=True)

# ===== Mismatch Rate (%) 추출 및 Global Mean 계산 =====
mismatch_rate_df = comparison_table_domain[
    comparison_table_domain.index.str.contains("Mismatch Rate")
].copy()

mismatch_rate_df["Domain"] = mismatch_rate_df.index.str.split(" - ").str[0]
mismatch_rate_df = mismatch_rate_df.rename(columns={
    "Old Prompt": "Original",
    "New Prompt": "Revised"
})[["Domain", "Original", "Revised"]]

# Global Mean 자동 계산 (마지막 행 추가)
global_mean_mismatch = mismatch_rate_df[["Original", "Revised"]].mean().to_frame().T
global_mean_mismatch.insert(0, "Domain", "Global Mean")
mismatch_rate_df = pd.concat([mismatch_rate_df, global_mean_mismatch], ignore_index=True)

# 결과 확인
ours_ratio_df, mismatch_rate_df

# shake rate는 별도로 
shake_rate_df = summary["domain_comparison"]
shake_rate_df.columns = ["Domain", "Original", "Revised"]
shake_rate_df[['Original', 'Revised']] = shake_rate_df[['Original', 'Revised']]*100
global_mean_shake = shake_rate_df[["Original", "Revised"]].mean().to_frame().T
global_mean_shake.insert(0, "Domain", "Global Mean")
shake_rate_df = pd.concat([shake_rate_df, global_mean_shake], ignore_index=True)

# %%
from matplotlib.ticker import PercentFormatter
import numpy as np

def _prepare_domain_data(df):
    """'Global Mean' 행이 있으면 제외하여 도메인 막대 데이터 구성,
       글로벌 평균은 항상 도메인 행들로부터 재계산."""
    df = df.copy()
    # 도메인 행만
    dom_df = df[df["Domain"].str.lower() != "global mean"].copy()
    # 글로벌 평균 재계산
    g_orig = dom_df["Original"].mean()
    g_rev  = dom_df["Revised"].mean()
    return dom_df, g_orig, g_rev

def plot_metric_by_domain(df, metric_title="Ours Win Rate (%)", savepath=None):
    """도메인별 Original/Revised 막대 + Global(Original/Revised) 수평선 시각화"""
    dom_df, g_orig, g_rev = _prepare_domain_data(df)

    x = np.arange(len(dom_df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width/2, dom_df["Original"].values, width, label="Original")
    ax.bar(x + width/2, dom_df["Revised"].values,  width, label="Revised")

    # 글로벌 기준선 (Original: dashed, Revised: solid)
    ln1 = ax.axhline(g_orig, linestyle="--", linewidth=1.5, label="Global (Original)")
    ln2 = ax.axhline(g_rev,  linestyle="-",  linewidth=1.5, label="Global (Revised)")

    ax.set_title(metric_title)
    ax.set_xticks(x)
    ax.set_xticklabels(dom_df["Domain"], rotation=0)
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))

    # 범례 (막대 2 + 기준선 2)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right", frameon=True)

    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    return fig, ax


plot_metric_by_domain(ours_ratio_df, "Ours Win Rate (%)")
plot_metric_by_domain(mismatch_rate_df, "Mismatch Rate (%)")
plot_metric_by_domain(shake_rate_df, "Shake Rate (%)")