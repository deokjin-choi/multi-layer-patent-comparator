import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def plot_strategy_radar(strategy_scores: list):
    aspects = list(strategy_scores[0]["scores"].keys())
    angles = np.linspace(0, 2 * np.pi, len(aspects), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10.colors

    ax.set_ylim(0, 5)
    ax.set_yticks(range(1, 6))
    ax.set_yticklabels([str(i) for i in range(1, 6)], color='gray')

    for i, entry in enumerate(strategy_scores):
        values = [entry["scores"][aspect]["score"] for aspect in aspects]
        values += values[:1]
        label = entry["patent_id"]
        lw = 3 if i == 0 else 1
        ax.plot(angles, values, color=colors[i % len(colors)], linewidth=lw, label=label)
        ax.scatter(angles, values, s=30, color=colors[i % len(colors)], zorder=3)

    ax.set_thetagrids(np.degrees(angles[:-1]), aspects)
    ax.spines['polar'].set_visible(True)
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)

    # ✅ 범례 위치와 크기 조정
    ax.legend(
        loc='lower right',
        bbox_to_anchor=(1.2, -0.1),
        fontsize='small',
        framealpha=0.6
    )

    plt.tight_layout()
    st.pyplot(fig)
