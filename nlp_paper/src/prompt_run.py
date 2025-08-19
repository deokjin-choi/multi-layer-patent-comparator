import sys, os, json, glob, importlib
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# === 경로 세팅 (실행 위치와 무관하게 __file__ 기준으로 설정) ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))           # nlp_paper/src
DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../data/prompts"))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../..")) # patent_compare
NLP_PAPER_SRC = CURRENT_DIR                                        # nlp_paper/src

sys.path.append(PROJECT_ROOT)   # app 모듈 불러오기 가능
sys.path.append(NLP_PAPER_SRC)  # metrics_eval 불러오기 가능

# === app 모듈 (운영 시스템 코드) ===
from app.controllers.engine import get_or_fetch_with_summary
from app.controllers.positioning import analyze_positioning

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

"""
Storage Path and Prompt Version Mapping

This section configures the save directory for experiment results.

- The actual prompt version (diff_v3 vs diff_v9) is determined inside
  `app.controllers.positioning.compare_positioning()`.
- External scripts (like this one) do not control the prompt version directly.
- Instead, we map those internal versions to storage folders:
    * diff_v3 → data/prompts/origin
    * diff_v9 → data/prompts/revised
- The `prompt_version` variable here is only used to choose the output folder
  (origin or revised), not to override the internal prompt selection.
- Trial results are saved as trial_1.csv ... trial_30.csv
- Aggregated results are saved as trial_full.csv
"""

def run_trials(prompt_version="diff_v9", trials=30):
    # "origin" 또는 "revised" 폴더에 저장하도록 설정
    save_dir = os.path.join(DATA_DIR, "origin" if prompt_version == "diff_v3" else "revised")
    os.makedirs(save_dir, exist_ok=True)

    for trial in tqdm(range(1, trials + 1), desc=f"Running {trials} Trials"):
        save_path = os.path.join(save_dir, f"trial_{trial}.csv")
        if os.path.exists(save_path):
            continue  # 이미 존재하면 스킵

        results = []
        for domain, ids in domain_patents.items():
            our_id = ids["ours"][0]
            comp_ids = ids["competitors"]

            our_patent = get_or_fetch_with_summary(our_id)
            competitor_patents = [get_or_fetch_with_summary(pid) for pid in comp_ids]

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

        df_trial = pd.DataFrame(results)
        df_trial.to_csv(save_path, index=False)

    # 전체 결과 통합
    all_csvs = glob.glob(os.path.join(save_dir, "trial_*.csv"))
    df_full = pd.concat([pd.read_csv(f) for f in all_csvs], ignore_index=True)
    df_full.to_csv(os.path.join(save_dir, "trial_full.csv"), index=False)


if __name__ == "__main__":
    run_trials(prompt_version="diff_v9", trials=30)