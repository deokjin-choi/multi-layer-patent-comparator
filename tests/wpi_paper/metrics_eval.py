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
        votes = [row["fr_winner"], row["tu_winner"], row["sv_winner"]]
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
        fr_vecs = model.encode(df["fr_reason"].fillna("").tolist(), show_progress_bar=False)
        tu_vecs = model.encode(df["tu_reason"].fillna("").tolist(), show_progress_bar=False)
        sv_vecs = model.encode(df["sv_reason"].fillna("").tolist(), show_progress_bar=False)
        overall_vecs = model.encode(df["overall_reason"].fillna("").tolist(), show_progress_bar=False)

        FR_mean = np.mean(fr_vecs, axis=0)
        TU_mean = np.mean(tu_vecs, axis=0)
        SV_mean = np.mean(sv_vecs, axis=0)

        fr_sim = cosine_similarity(overall_vecs, [FR_mean]).flatten()
        tu_sim = cosine_similarity(overall_vecs, [TU_mean]).flatten()
        sv_sim = cosine_similarity(overall_vecs, [SV_mean]).flatten()

        dominant = []
        for i in range(len(fr_sim)):
            scores = {"FR": fr_sim[i], "TU": tu_sim[i], "SV": sv_sim[i]}
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

        return {
            "overall_ours_ratio": (df["overall_winner"] == "ours").mean() * 100,
            "domain_ours_ratio": df.groupby("domain")["overall_winner"].apply(lambda x: (x == "ours").mean() * 100).to_dict(),
            "axis_ours_ratio": {col: (df[col] == "ours").mean() * 100 for col in ["fr_winner", "tu_winner", "sv_winner"]},
            "accuracy_kappa": {
                col: {
                    "accuracy": accuracy_score(df[col], df["overall_winner"]),
                    "kappa": cohen_kappa_score(df[col], df["overall_winner"])
                }
                for col in ["fr_winner", "tu_winner", "sv_winner"]
            },
            "mismatch_rate": (~df["match"]).mean() * 100,
            "mismatch_distribution": df.loc[~df["match"], "overall_winner"].value_counts(normalize=True).mul(100).to_dict(),
            "domain_mismatch_rate": domain_mismatch_rate,
            "domain_mismatch_distribution": domain_mismatch_distribution,
            "dominant_reason_global": df["reason_dominant_aspect"].value_counts(normalize=True).mul(100).to_dict(),
            "cohesion_scores": {
                "FR": cohesion_score(fr_vecs),
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
        "Dominant Reason = FR (%)": [
            old_result["dominant_reason_global"].get("FR", 0),
            new_result["dominant_reason_global"].get("FR", 0),
        ],
        "Dominant Reason = TU (%)": [
            old_result["dominant_reason_global"].get("TU", 0),
            new_result["dominant_reason_global"].get("TU", 0),
        ],
        "Dominant Reason = SV (%)": [
            old_result["dominant_reason_global"].get("SV", 0),
            new_result["dominant_reason_global"].get("SV", 0),
        ],
        "FR Accuracy": [
            old_result["accuracy_kappa"]["fr_winner"]["accuracy"],
            new_result["accuracy_kappa"]["fr_winner"]["accuracy"],
        ],
        "FR Kappa": [
            old_result["accuracy_kappa"]["fr_winner"]["kappa"],
            new_result["accuracy_kappa"]["fr_winner"]["kappa"],
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
        "FR Cohesion": [
            old_result["cohesion_scores"]["FR"],
            new_result["cohesion_scores"]["FR"],
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


def compare_results_table(old_result, new_result):
    comparison = {
        "Overall Ours Ratio (%)": [old_result["overall_ours_ratio"], new_result["overall_ours_ratio"]],
        "Mismatch Rate (%)": [old_result["mismatch_rate"], new_result["mismatch_rate"]],
        "FR Accuracy": [old_result["accuracy_kappa"]["fr_winner"]["accuracy"], new_result["accuracy_kappa"]["fr_winner"]["accuracy"]],
        "FR Kappa": [old_result["accuracy_kappa"]["fr_winner"]["kappa"], new_result["accuracy_kappa"]["fr_winner"]["kappa"]],
        "TU Accuracy": [old_result["accuracy_kappa"]["tu_winner"]["accuracy"], new_result["accuracy_kappa"]["tu_winner"]["accuracy"]],
        "TU Kappa": [old_result["accuracy_kappa"]["tu_winner"]["kappa"], new_result["accuracy_kappa"]["tu_winner"]["kappa"]],
        "SV Accuracy": [old_result["accuracy_kappa"]["sv_winner"]["accuracy"], new_result["accuracy_kappa"]["sv_winner"]["accuracy"]],
        "SV Kappa": [old_result["accuracy_kappa"]["sv_winner"]["kappa"], new_result["accuracy_kappa"]["sv_winner"]["kappa"]],
        "FR Cohesion": [old_result["cohesion_scores"]["FR"], new_result["cohesion_scores"]["FR"]],
        "TU Cohesion": [old_result["cohesion_scores"]["TU"], new_result["cohesion_scores"]["TU"]],
        "SV Cohesion": [old_result["cohesion_scores"]["SV"], new_result["cohesion_scores"]["SV"]],
    }

    # 1. 도메인별 ours 비율 비교
    all_domains = set(old_result["domain_ours_ratio"].keys()) | set(new_result["domain_ours_ratio"].keys())
    for domain in sorted(all_domains):
        old_val = old_result["domain_ours_ratio"].get(domain, np.nan)
        new_val = new_result["domain_ours_ratio"].get(domain, np.nan)
        comparison[f"Domain '{domain}' Ours Ratio (%)"] = [old_val, new_val]

    # 2. Mismatch에 한한 최종 승자 비율 비교
    all_mismatch_keys = set(old_result["mismatch_distribution"].keys()) | set(new_result["mismatch_distribution"].keys())
    for key in sorted(all_mismatch_keys):
        old_val = old_result["mismatch_distribution"].get(key, 0)
        new_val = new_result["mismatch_distribution"].get(key, 0)
        comparison[f"Mismatch Winner = {key} (%)"] = [old_val, new_val]

    # 3. 도메인별 mismatch 비율 및 승자 비율
    if "domain_mismatch_stats" in old_result and "domain_mismatch_stats" in new_result:
        all_domains_mismatch = set(old_result["domain_mismatch_stats"].keys()) | set(new_result["domain_mismatch_stats"].keys())
        for domain in sorted(all_domains_mismatch):
            old_stats = old_result["domain_mismatch_stats"].get(domain, {})
            new_stats = new_result["domain_mismatch_stats"].get(domain, {})

            old_mis_rate = old_stats.get("mismatch_rate", np.nan)
            new_mis_rate = new_stats.get("mismatch_rate", np.nan)
            comparison[f"Domain '{domain}' Mismatch Rate (%)"] = [old_mis_rate, new_mis_rate]

            all_winners = set(old_stats.get("winner_distribution", {}).keys()) | set(new_stats.get("winner_distribution", {}).keys())
            for winner in sorted(all_winners):
                old_w = old_stats.get("winner_distribution", {}).get(winner, 0)
                new_w = new_stats.get("winner_distribution", {}).get(winner, 0)
                comparison[f"Domain '{domain}' Winner = {winner} (Mismatch, %)]"] = [old_w, new_w]

    df_comp = pd.DataFrame(comparison, index=["Old Prompt", "New Prompt"]).T
    return df_comp



# Function to plot a bar chart comparing old and new results
def plot_comparison_bar_chart(old_result, new_result):
    # 비교 항목 및 값
    metrics = {
        "Ours Ratio": [old_result["overall_ours_ratio"], new_result["overall_ours_ratio"]],
        "Mismatch Rate": [old_result["mismatch_rate"], new_result["mismatch_rate"]],
        "FR Accuracy": [old_result["accuracy_kappa"]["fr_winner"]["accuracy"], new_result["accuracy_kappa"]["fr_winner"]["accuracy"]],
        "TU Accuracy": [old_result["accuracy_kappa"]["tu_winner"]["accuracy"], new_result["accuracy_kappa"]["tu_winner"]["accuracy"]],
        "SV Accuracy": [old_result["accuracy_kappa"]["sv_winner"]["accuracy"], new_result["accuracy_kappa"]["sv_winner"]["accuracy"]],
    }

    labels = list(metrics.keys())
    old_values = [v[0] for v in metrics.values()]
    new_values = [v[1] for v in metrics.values()]

    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x, old_values, width, label='Old Prompt')
    plt.bar([i + width for i in x], new_values, width, label='New Prompt')
    plt.xticks([i + width/2 for i in x], labels, rotation=45)
    plt.ylabel("Score / %")
    plt.title("Prompt Performance Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()


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




