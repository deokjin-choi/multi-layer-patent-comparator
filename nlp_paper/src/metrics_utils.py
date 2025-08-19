import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from collections import Counter
from sentence_transformers import SentenceTransformer


# Define the main evaluation function
def evaluate_comparison_results(old_df, new_df):

    def majority_vote(row):
        votes = [row["fp_winner"], row["tu_winner"], row["sv_winner"]]
        count = Counter(votes)
        top_count = count.most_common(1)[0][1]
        top_winners = [k for k, v in count.items() if v == top_count]
        return "tie" if len(top_winners) > 1 else top_winners[0]

    def cohesion_score(vectors):
        center = np.mean(vectors, axis=0)
        dists = pairwise_distances(vectors, [center], metric="cosine")
        return dists.mean()

    def process_df(df):
        model = SentenceTransformer("AI-Growth-Lab/PatentSBerta")

        df["majority_winner"] = df.apply(majority_vote, axis=1)
        df["match"] = df["overall_winner"] == df["majority_winner"]

        # 임베딩
        fp_vecs = model.encode(df["fp_reason"].fillna("").tolist(), show_progress_bar=False)
        tu_vecs = model.encode(df["tu_reason"].fillna("").tolist(), show_progress_bar=False)
        sv_vecs = model.encode(df["sv_reason"].fillna("").tolist(), show_progress_bar=False)
        overall_vecs = model.encode(df["overall_reason"].fillna("").tolist(), show_progress_bar=False)

        FP_mean = np.mean(fp_vecs, axis=0)
        TU_mean = np.mean(tu_vecs, axis=0)
        SV_mean = np.mean(sv_vecs, axis=0)

        fp_sim = cosine_similarity(overall_vecs, [FP_mean]).flatten()
        tu_sim = cosine_similarity(overall_vecs, [TU_mean]).flatten()
        sv_sim = cosine_similarity(overall_vecs, [SV_mean]).flatten()

        dominant = []
        for i in range(len(fp_sim)):
            scores = {"FP": fp_sim[i], "TU": tu_sim[i], "SV": sv_sim[i]}
            dominant.append(max(scores, key=scores.get))
        df["reason_dominant_aspect"] = dominant

        # 도메인별 mismatch 계산
        domain_mismatch_rate = df.groupby("domain")["match"].apply(lambda x: (~x).mean() * 100).to_dict()
        domain_mismatch_distribution = (
            df[~df["match"]]
            .groupby("domain")["overall_winner"]
            .value_counts(normalize=True)
            .mul(100)
            .unstack(fill_value=0)
            .to_dict(orient="index")
        )

        # 도메인별 accuracy / kappa 계산
        accuracy_kappa_domain = {
            domain: {
                col: {
                    "accuracy": accuracy_score(gr[col], gr["overall_winner"]),
                    "kappa": cohen_kappa_score(gr[col], gr["overall_winner"])
                }
                for col in ["fp_winner", "tu_winner", "sv_winner"]
            }
            for domain, gr in df.groupby("domain")
        }

        # 도메인별 dominant reason 비율
        dominant_reason_domain = (
            df.groupby("domain")["reason_dominant_aspect"]
            .value_counts(normalize=True)
            .mul(100)
            .unstack(fill_value=0)
            .to_dict(orient="index")
        )

        return {
            "overall_ours_ratio": (df["overall_winner"] == "ours").mean() * 100,
            "domain_ours_ratio": df.groupby("domain")["overall_winner"].apply(lambda x: (x == "ours").mean() * 100).to_dict(),
            "axis_ours_ratio": {col: (df[col] == "ours").mean() * 100 for col in ["fp_winner", "tu_winner", "sv_winner"]},
            "accuracy_kappa": {
                col: {
                    "accuracy": accuracy_score(df[col], df["overall_winner"]),
                    "kappa": cohen_kappa_score(df[col], df["overall_winner"])
                }
                for col in ["fp_winner", "tu_winner", "sv_winner"]
            },
            "accuracy_kappa_domain": accuracy_kappa_domain,
            "mismatch_rate": (~df["match"]).mean() * 100,
            "mismatch_distribution": df.loc[~df["match"], "overall_winner"].value_counts(normalize=True).mul(100).to_dict(),
            "domain_mismatch_rate": domain_mismatch_rate,
            "domain_mismatch_distribution": domain_mismatch_distribution,
            "dominant_reason_global": df["reason_dominant_aspect"].value_counts(normalize=True).mul(100).to_dict(),
            "dominant_reason_domain": dominant_reason_domain,
            "cohesion_scores": {
                "FP": cohesion_score(fp_vecs),
                "TU": cohesion_score(tu_vecs),
                "SV": cohesion_score(sv_vecs)
            },
        }

    result_old = process_df(old_df)
    result_new = process_df(new_df)
    return result_old, result_new


def compare_results_overall_table(old_result, new_result):
    comparison = {
        "Overall Ours Ratio (%)": [old_result["overall_ours_ratio"], new_result["overall_ours_ratio"]],
        "Overall Mismatch Rate (%)": [old_result["mismatch_rate"], new_result["mismatch_rate"]],
        "-Mismatch Winner = ours (%)": [
            old_result["mismatch_distribution"].get("ours", 0),
            new_result["mismatch_distribution"].get("ours", 0),
        ],
        "-Mismatch Winner = competitor (%)": [
            old_result["mismatch_distribution"].get("competitor", 0),
            new_result["mismatch_distribution"].get("competitor", 0),
        ],
        "-Mismatch Winner = tie (%)": [
            old_result["mismatch_distribution"].get("tie", 0),
            new_result["mismatch_distribution"].get("tie", 0),
        ],
         # ⬇⬇ 추가 시작 ⬇⬇
        "Dominant Reason = FP (%)": [
            old_result["dominant_reason_global"].get("FP", 0),
            new_result["dominant_reason_global"].get("FP", 0),
        ],
        "Dominant Reason = TU (%)": [
            old_result["dominant_reason_global"].get("TU", 0),
            new_result["dominant_reason_global"].get("TU", 0),
        ],
        "Dominant Reason = SV (%)": [
            old_result["dominant_reason_global"].get("SV", 0),
            new_result["dominant_reason_global"].get("SV", 0),
        ],
        "FP Accuracy": [
            old_result["accuracy_kappa"]["fp_winner"]["accuracy"],
            new_result["accuracy_kappa"]["fp_winner"]["accuracy"],
        ],
        "FP Kappa": [
            old_result["accuracy_kappa"]["fp_winner"]["kappa"],
            new_result["accuracy_kappa"]["fp_winner"]["kappa"],
        ],
        "TU Accuracy": [
            old_result["accuracy_kappa"]["tu_winner"]["accuracy"],
            new_result["accuracy_kappa"]["tu_winner"]["accuracy"],
        ],
        "TU Kappa": [
            old_result["accuracy_kappa"]["tu_winner"]["kappa"],
            new_result["accuracy_kappa"]["tu_winner"]["kappa"],
        ],
        "SV Accuracy": [
            old_result["accuracy_kappa"]["sv_winner"]["accuracy"],
            new_result["accuracy_kappa"]["sv_winner"]["accuracy"],
        ],
        "SV Kappa": [
            old_result["accuracy_kappa"]["sv_winner"]["kappa"],
            new_result["accuracy_kappa"]["sv_winner"]["kappa"],
        ],
        "FP Cohesion": [
            old_result["cohesion_scores"]["FP"],
            new_result["cohesion_scores"]["FP"],
        ],
        "TU Cohesion": [
            old_result["cohesion_scores"]["TU"],
            new_result["cohesion_scores"]["TU"],
        ],
        "SV Cohesion": [
            old_result["cohesion_scores"]["SV"],
            new_result["cohesion_scores"]["SV"],
        ],
    }

    df_comp = pd.DataFrame(comparison, index=["Old Prompt", "New Prompt"]).T
    return df_comp


def compare_detailed_results_by_domain(old_result, new_result):
    domains = old_result["domain_ours_ratio"].keys()
    comparison = {}

    for domain in domains:
        # Ours ratio, mismatch rate, mismatch winner
        comparison[f"{domain} - Ours Ratio (%)"] = [
            old_result["domain_ours_ratio"].get(domain, 0),
            new_result["domain_ours_ratio"].get(domain, 0),
        ]
        comparison[f"{domain} - Mismatch Rate (%)"] = [
            old_result["domain_mismatch_rate"].get(domain, 0),
            new_result["domain_mismatch_rate"].get(domain, 0),
        ]
        for label in ["ours", "competitor", "tie"]:
            comparison[f"{domain} - Mismatch Winner = {label} (%)"] = [
                old_result["domain_mismatch_distribution"].get(domain, {}).get(label, 0),
                new_result["domain_mismatch_distribution"].get(domain, {}).get(label, 0),
            ]
        # Dominant reason
        for axis in ["FP", "TU", "SV"]:
            comparison[f"{domain} - Dominant Reason = {axis} (%)"] = [
                old_result.get("dominant_reason_domain", {}).get(domain, {}).get(axis, 0),
                new_result.get("dominant_reason_domain", {}).get(domain, {}).get(axis, 0),
            ]
        # Accuracy / Kappa
        for axis in ["fp_winner", "tu_winner", "sv_winner"]:
            label = axis.upper().replace("_WINNER", "")
            comparison[f"{domain} - {label} Accuracy"] = [
                old_result.get("accuracy_kappa_domain", {}).get(domain, {}).get(axis, {}).get("accuracy", 0),
                new_result.get("accuracy_kappa_domain", {}).get(domain, {}).get(axis, {}).get("accuracy", 0),
            ]
            comparison[f"{domain} - {label} Kappa"] = [
                old_result.get("accuracy_kappa_domain", {}).get(domain, {}).get(axis, {}).get("kappa", 0),
                new_result.get("accuracy_kappa_domain", {}).get(domain, {}).get(axis, {}).get("kappa", 0),
            ]

    df_detailed_domain = pd.DataFrame(comparison, index=["Old Prompt", "New Prompt"]).T
    return df_detailed_domain


def compute_shakiness(df):
    # 조합별 흔들림 계산
    def shakiness(group):
        winners = group.sort_values("trial")["overall_winner"].tolist()
        num_changes = sum(w1 != w2 for w1, w2 in zip(winners, winners[1:]))
        rate = num_changes / (len(winners) - 1) if len(winners) > 1 else 0
        return pd.Series({
            "shake_count": num_changes,
            "shake_rate": rate,
            "n_trials": len(winners),
            "domain": group["domain"].iloc[0],
        })

    # 조합별 흔들림 계산
    shake_per_pair = df.groupby(["our_id", "competitor_id"]).apply(shakiness).reset_index()

    # 도메인별 흔들림 평균
    shake_by_domain = (
        shake_per_pair.groupby("domain")["shake_rate"]
        .mean()
        .reset_index(name="avg_shake_rate")
    )

    # 전체 흔들림 평균
    overall_avg = shake_per_pair["shake_rate"].mean()

    return {
        "pair": shake_per_pair,
        "domain": shake_by_domain,
        "overall": overall_avg
    }


def compare_shakiness_summary(df_old, df_new):
    result_old = compute_shakiness(df_old)
    result_new = compute_shakiness(df_new)

    # 전체 평균 흔들림 비교
    overall_df = pd.DataFrame({
        "metric": ["overall_avg_shake"],
        "old": [result_old["overall"]],
        "new": [result_new["overall"]]
    })

    # 도메인별 비교
    domain_df = pd.merge(
        result_old["domain"], result_new["domain"],
        on="domain", how="outer", suffixes=('_old', '_new')
    ).fillna(0)

    # 조합별 비교 (optional, 필요 시 활용)
    pair_df = pd.merge(
        result_old["pair"][["our_id", "competitor_id", "shake_rate"]],
        result_new["pair"][["our_id", "competitor_id", "shake_rate"]],
        on=["our_id", "competitor_id"],
        how="outer", suffixes=('_old', '_new')
    ).fillna(0)

    return {
        "overall_comparison": overall_df,
        "domain_comparison": domain_df,
        "pair_comparison": pair_df
    }




