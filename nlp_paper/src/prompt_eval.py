"""
Prompt Evaluation Script (origin vs revised)

This script loads previously generated prompt-run results and computes
evaluation metrics and visualizations comparing:
    - origin (diff_v3)  vs  revised (diff_v9)

Key points:
- The actual prompt version mapping (diff_v3 → origin, diff_v9 → revised)
  is enforced in the prompt-run step and by directory layout:
    * data/prompts/origin/trial_full.csv
    * data/prompts/revised/trial_full.csv
- This script does NOT run LLM; it ONLY:
    1) loads trial_full.csv files
    2) computes metrics using `metrics_eval.py`
    3) prints comparison tables
    4) plots distributions and domain-wise bars (optionally saves figures)

Usage:
    python prompt_eval.py \
        --origin  ../data/prompts/origin/trial_full.csv \
        --revised ../data/prompts/revised/trial_full.csv \
        --outdir  ../data/prompts/figs

If --outdir is omitted, figures are just shown (not saved).
"""

import sys, os, json, glob, importlib, argparse
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# === Path setup (stable regardless of working directory) ===
CURRENT_DIR   = os.path.dirname(os.path.abspath(__file__))         # nlp_paper/src
DATA_DIR      = os.path.abspath(os.path.join(CURRENT_DIR, "../data/prompts"))
PROJECT_ROOT  = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))  # patent_compare
NLP_PAPER_SRC = CURRENT_DIR

sys.path.append(PROJECT_ROOT)   # to import `app` if needed later
sys.path.append(NLP_PAPER_SRC)  # to import local src modules like metrics_eval

# === Experimental metrics module ===
import metrics_utils

from metrics_utils import (
    evaluate_comparison_results,
    compare_results_overall_table,
    compare_detailed_results_by_domain,
    compare_shakiness_summary,
)

importlib.reload(metrics_utils)

# ---------- Helper functions (kept as-is / slightly organized) ----------

def get_winner_distribution(df, column):
    """Compute ours/competitor/tie percentage for a given winner column."""
    dist = df[column].value_counts(normalize=True) * 100
    return {
        'ours':       float(dist.get('ours', 0)),
        'competitor': float(dist.get('competitor', 0)),
        'tie':        float(dist.get('tie', 0)),
    }

def prepare_all_distributions(old_df, new_df):
    """Compute winner distributions for FP, TU, SV (old vs new)."""
    return {
        'FP': (get_winner_distribution(old_df, 'fp_winner'),
               get_winner_distribution(new_df, 'fp_winner')),
        'TU': (get_winner_distribution(old_df, 'tu_winner'),
               get_winner_distribution(new_df, 'tu_winner')),
        'SV': (get_winner_distribution(old_df, 'sv_winner'),
               get_winner_distribution(new_df, 'sv_winner')),
    }

def plot_winner_distributions(distributions, savepath=None):
    """
    Plot stacked bars of winner distributions (ours/competitor/tie) for FP/TU/SV.
    If savepath is given, saves the figure; otherwise, shows it.
    """
    labels = ['Original Prompt', 'Revised Prompt']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (dimension, (old_dist, new_dist)) in enumerate(distributions.items()):
        ours = [old_dist['ours'], new_dist['ours']]
        comp = [old_dist['competitor'], new_dist['competitor']]
        tie  = [old_dist['tie'], new_dist['tie']]

        bottoms1 = ours
        bottoms2 = [i + j for i, j in zip(ours, comp)]

        ax = axes[idx]
        bars1 = ax.bar(labels, ours, label='Ours')
        bars2 = ax.bar(labels, comp, bottom=bottoms1, label='Competitor')
        bars3 = ax.bar(labels, tie,  bottom=bottoms2, label='Tie')

        # percentage labels
        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h/2, f'{h:.1f}%', ha='center', va='center', fontsize=9)
        for bar, base in zip(bars2, bottoms1):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, base + h/2, f'{h:.1f}%', ha='center', va='center', fontsize=9)
        for bar, base in zip(bars3, bottoms2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, base + h/2, f'{h:.1f}%', ha='center', va='center', fontsize=9)

        ax.set_ylim(0, 100)
        ax.set_ylabel('Percentage (%)')
        ax.set_title(f'{dimension} Winner Distribution (Percentage)')
        ax.legend(loc='lower left', fontsize=8)

    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

from matplotlib.ticker import PercentFormatter
import numpy as np

def _prepare_domain_data(df):
    """Remove 'Global Mean' row (if exists) and recompute global means."""
    df = df.copy()
    dom_df = df[df["Domain"].str.lower() != "global mean"].copy()
    g_orig = float(dom_df["Original"].mean())
    g_rev  = float(dom_df["Revised"].mean())
    return dom_df, g_orig, g_rev

def plot_metric_by_domain(df, metric_title="Ours Win Rate (%)", savepath=None):
    """
    Bar chart per domain for Original/Revised with horizontal global mean lines.
    If savepath is provided, saves the figure; otherwise shows it.
    """
    dom_df, g_orig, g_rev = _prepare_domain_data(df)

    x = np.arange(len(dom_df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width/2, dom_df["Original"].values, width, label="Original")
    ax.bar(x + width/2, dom_df["Revised"].values,  width, label="Revised")

    ax.axhline(g_orig, linestyle="--", linewidth=1.5, label="Global (Original)")
    ax.axhline(g_rev,  linestyle="-",  linewidth=1.5, label="Global (Revised)")

    ax.set_title(metric_title)
    ax.set_xticks(x)
    ax.set_xticklabels(dom_df["Domain"], rotation=0)
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)

    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return fig, ax

# ---------- Main evaluation pipeline ----------

def evaluate_and_plot(origin_csv, revised_csv, outdir=None):
    """
    Load trial_full.csv for origin/revised, compute metrics and generate plots.
    If outdir is provided, save figures there; otherwise display interactively.
    """
    # 1) Load data
    old_df = pd.read_csv(origin_csv)
    new_df = pd.read_csv(revised_csv)

    # 2) Core metrics via metrics_eval
    old_result, new_result = evaluate_comparison_results(old_df, new_df)
    comparison_table = compare_results_overall_table(old_result, new_result)
    print("\n=== Overall Comparison Table ===")
    print(comparison_table)

    summary = compare_shakiness_summary(old_df, new_df)
    print("\n=== Shakiness Summary (Overall) ===")
    print(summary["overall_comparison"])
    print("\n=== Shakiness Summary (By Domain) ===")
    print(summary["domain_comparison"])

    # 3) Winner distributions (FP/TU/SV)
    distributions = prepare_all_distributions(old_df, new_df)
    dist_fig_path = os.path.join(outdir, "winner_distributions.png") if outdir else None
    plot_winner_distributions(distributions, savepath=dist_fig_path)

    # 4) Domain-level tables
    comparison_table_domain = compare_detailed_results_by_domain(old_result, new_result)

    # Ours Ratio (%)
    ours_ratio_df = comparison_table_domain[
        comparison_table_domain.index.str.contains("Ours Ratio")
    ].copy()
    ours_ratio_df["Domain"] = ours_ratio_df.index.str.split(" - ").str[0]
    ours_ratio_df = ours_ratio_df.rename(columns={
        "Old Prompt": "Original",
        "New Prompt": "Revised"
    })[["Domain", "Original", "Revised"]]
    global_mean_ours = ours_ratio_df[["Original", "Revised"]].mean().to_frame().T
    global_mean_ours.insert(0, "Domain", "Global Mean")
    ours_ratio_df = pd.concat([ours_ratio_df, global_mean_ours], ignore_index=True)

    # Mismatch Rate (%)
    mismatch_rate_df = comparison_table_domain[
        comparison_table_domain.index.str.contains("Mismatch Rate")
    ].copy()
    mismatch_rate_df["Domain"] = mismatch_rate_df.index.str.split(" - ").str[0]
    mismatch_rate_df = mismatch_rate_df.rename(columns={
        "Old Prompt": "Original",
        "New Prompt": "Revised"
    })[["Domain", "Original", "Revised"]]
    global_mean_mismatch = mismatch_rate_df[["Original", "Revised"]].mean().to_frame().T
    global_mean_mismatch.insert(0, "Domain", "Global Mean")
    mismatch_rate_df = pd.concat([mismatch_rate_df, global_mean_mismatch], ignore_index=True)

    # Shake Rate (%)
    shake_rate_df = summary["domain_comparison"].copy()
    shake_rate_df.columns = ["Domain", "Original", "Revised"]
    shake_rate_df[["Original", "Revised"]] = shake_rate_df[["Original", "Revised"]] * 100
    global_mean_shake = shake_rate_df[["Original", "Revised"]].mean().to_frame().T
    global_mean_shake.insert(0, "Domain", "Global Mean")
    shake_rate_df = pd.concat([shake_rate_df, global_mean_shake], ignore_index=True)

    # 5) Plots per domain
    ours_path     = os.path.join(outdir, "ours_ratio_by_domain.png")     if outdir else None
    mismatch_path = os.path.join(outdir, "mismatch_rate_by_domain.png")  if outdir else None
    shake_path    = os.path.join(outdir, "shake_rate_by_domain.png")     if outdir else None

    plot_metric_by_domain(ours_ratio_df,     "Ours Win Rate (%)", savepath=ours_path)
    plot_metric_by_domain(mismatch_rate_df,  "Mismatch Rate (%)", savepath=mismatch_path)
    plot_metric_by_domain(shake_rate_df,     "Shake Rate (%)",    savepath=shake_path)

    return {
        "comparison_table": comparison_table,
        "ours_ratio_df": ours_ratio_df,
        "mismatch_rate_df": mismatch_rate_df,
        "shake_rate_df": shake_rate_df,
    }

# ---------- CLI entrypoint ----------

def build_parser():
    origin_default  = os.path.join(DATA_DIR, "origin",  "trial_full.csv")
    revised_default = os.path.join(DATA_DIR, "revised", "trial_full.csv")
    outdir_default  = os.path.join(DATA_DIR, "figs")

    p = argparse.ArgumentParser(
        description="Evaluate prompt results (origin vs revised) and plot metrics."
    )
    p.add_argument("--origin",  type=str, default=origin_default,
                   help="Path to origin (diff_v3) trial_full.csv")
    p.add_argument("--revised", type=str, default=revised_default,
                   help="Path to revised (diff_v9) trial_full.csv")
    p.add_argument("--outdir",  type=str, default=outdir_default,
                   help="Directory to save figures if --save is used")
    p.add_argument("--save",    action="store_true",
                   help="If set, save figures to --outdir; default is show-only.")
    return p

if __name__ == "__main__":
    parser = build_parser()

    # Jupyter/Notebook에서는 커널이 붙이는 알 수 없는 인자(--f=...)가 있음 → 무시
    # CLI에선 일반적으로 전체 argv 파싱
    if "ipykernel_launcher" in sys.argv[0]:
        args, _ = parser.parse_known_args([])   # 기본값만 사용 (show-only)
    else:
        args, _ = parser.parse_known_args()     # CLI 인자 사용

    outdir = args.outdir if args.save else None
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    evaluate_and_plot(args.origin, args.revised, outdir=outdir)