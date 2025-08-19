#!/usr/bin/env python
import os, json, pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
NLP_PAPER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 경로
ORIGIN_CSV   = os.path.join(NLP_PAPER_DIR, "data/prompts/origin/trial_full.csv")
REVISED_CSV  = os.path.join(NLP_PAPER_DIR, "data/prompts/revised/trial_full.csv")
JUSTIF_DIR   = os.path.join(NLP_PAPER_DIR, "data/peft/justifications")
PROBS_DIR    = os.path.join(NLP_PAPER_DIR, "data/peft/token_probs")
FIG_DIR      = os.path.join(NLP_PAPER_DIR, "data/peft/figs")
BASE_MODEL_PATH = "/mnt/d/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/7231864981174d9bee8c7687c24c8344414eae6b"

def load_and_merge_data(origin_csv, revised_csv):
    d1 = pd.read_csv(origin_csv)
    d2 = pd.read_csv(revised_csv)
    return pd.concat([d1, d2], ignore_index=True)

def add_dominant_reason(df, df_valid, model_name="AI-Growth-Lab/PatentSBerta"):
    model = SentenceTransformer(model_name)

    fp_vecs = model.encode(df["fp_reason"].fillna("").tolist(), show_progress_bar=True)
    tu_vecs = model.encode(df["tu_reason"].fillna("").tolist(), show_progress_bar=True)
    sv_vecs = model.encode(df["sv_reason"].fillna("").tolist(), show_progress_bar=True)
    overall_vecs = model.encode(df_valid["overall_reason"].fillna("").tolist(), show_progress_bar=True)

    FP_mean = np.mean(fp_vecs, axis=0)
    TU_mean = np.mean(tu_vecs, axis=0)
    SV_mean = np.mean(sv_vecs, axis=0)

    fp_sim = cosine_similarity(overall_vecs, [FP_mean]).flatten()
    tu_sim = cosine_similarity(overall_vecs, [TU_mean]).flatten()
    sv_sim = cosine_similarity(overall_vecs, [SV_mean]).flatten()

    dominant = []   
    for i in range(len(fp_sim)):
        sim_dict = {"FP": fp_sim[i], "TU": tu_sim[i], "SV": sv_sim[i]}
        dominant.append(max(sim_dict, key=sim_dict.get))
    
    df_valid["reason_dominant_aspect"] = dominant
    return df_valid

def compute_mean_embeddings(df_total, model_name="AI-Growth-Lab/PatentSBerta"):
    model = SentenceTransformer(model_name)
    fp_vecs = model.encode(df_total["fp_reason"].fillna("").tolist(), show_progress_bar=True)
    tu_vecs = model.encode(df_total["tu_reason"].fillna("").tolist(), show_progress_bar=True)
    sv_vecs = model.encode(df_total["sv_reason"].fillna("").tolist(), show_progress_bar=True)

    FP_mean = np.mean(fp_vecs, axis=0)
    TU_mean = np.mean(tu_vecs, axis=0)
    SV_mean = np.mean(sv_vecs, axis=0)

    return model, FP_mean, TU_mean, SV_mean

# tokenizer.tokenize 함수
# -키워드 토큰화 : "cardiovascular disease monitoring" 
#  -> ["cardio", "_vascular", "_disease", "monit","_oring"]    
def tokenize_keywords(tokenizer, phrase_str):
    phrases = [p.strip() for p in str(phrase_str).split(',') if p.strip()]
    toks = []
    for phrase in phrases:
        toks.extend(tokenizer.tokenize(phrase))
    return toks

def build_token_prob_map_avg(prob_dicts, tokenizer):
    token_sums = defaultdict(float)
    token_counts = defaultdict(int)

    # 각 행별 key(토큰 id)와 value(확률)를 저장
    for prob_dict in prob_dicts:
        for token_id, prob in prob_dict.items():

            # 토큰 id를 토큰 문자열로 변환 : 
            # [101, 200, 300] -> ["[CLS]", "cardio", "_vascular", "_disease", "monit","_oring", "[SEP]"]
            token_str = tokenizer.convert_ids_to_tokens(token_id)
            token_sums[token_str] += prob
            token_counts[token_str] += 1

    return {token: token_sums[token] / token_counts[token] for token in token_sums}


def compute_delta(tokens, base_map, fine_map):
    deltas = []
    for token in tokens:
        base_p = base_map.get(token, 0.0)
        fine_p = fine_map.get(token, 0.0)
        deltas.append(fine_p - base_p)
    return np.mean(deltas) if deltas else 0.0

def load_data_and_preprocess():
    """데이터를 로드하고 필요한 전처리를 수행합니다."""
    # 1) 로드
    df_total = load_and_merge_data(ORIGIN_CSV, REVISED_CSV)
    base_outputs = pd.read_csv(os.path.join(JUSTIF_DIR, "base_outputs.csv"))
    ft_outputs = pd.read_csv(os.path.join(JUSTIF_DIR, "finetuned_outputs.csv"))
    with open(os.path.join(PROBS_DIR, "base_probs.pkl"), "rb") as f:
        base_probs = pickle.load(f)
    with open(os.path.join(PROBS_DIR, "finetuned_probs.pkl"), "rb") as f:
        finetuned_probs = pickle.load(f)

    # base_finetuned_keyword.txt(chat gpt로 생성)
    kw_path = os.path.join(JUSTIF_DIR, "base_finetuned_keyword.txt")
    if os.path.exists(kw_path):
        with open(kw_path, "r", encoding="utf-8") as f:
            keyword_lines = [line.strip() if line.strip() else np.nan for line in f]
        df_keywords = pd.DataFrame({"keyword": keyword_lines})
    else:
        df_keywords = pd.DataFrame({"keyword": [np.nan] * len(base_outputs)})
    
    base_outputs = base_outputs.join(df_keywords)
    base_outputs_notna = base_outputs[base_outputs['keyword'].notna()]
    ft_outputs = ft_outputs.join(df_keywords)

    base_outputs_notna = add_dominant_reason(df_total, base_outputs_notna, model_name="AI-Growth-Lab/PatentSBerta")
    ft_outputs = add_dominant_reason(df_total, ft_outputs, model_name="AI-Growth-Lab/PatentSBerta")

    # Reason Dominant Aspect 분포 출력
    print("Base Model Dominant Aspect Distribution:")
    print(base_outputs_notna['reason_dominant_aspect'].value_counts(normalize=True))
    print("\nFine-tuned Model Dominant Aspect Distribution:")
    print(ft_outputs['reason_dominant_aspect'].value_counts(normalize=True))
    
    return df_total, base_outputs, ft_outputs, df_keywords, base_probs, finetuned_probs

def process_keywords_and_join(df_total, base_outputs, ft_outputs, df_keywords):
    """키워드 분해 및 데이터프레임 병합을 처리합니다."""
    # 이름 정리 및 병합
    base_outputs = base_outputs.rename(columns={"overall_reason": "base_reason"})
    ft_outputs = ft_outputs.rename(columns={"overall_reason": "finetuned_reason"})
    df_join = base_outputs.merge(ft_outputs, on="Index", how="outer").merge(df_keywords, left_on="Index", right_index=True, how="left")
    df_join = df_join[["Index", "base_reason", "finetuned_reason", "keyword"]]
    
    # 2) 임베딩 평균
    model, FP_mean, TU_mean, SV_mean = compute_mean_embeddings(df_total, "AI-Growth-Lab/PatentSBerta")
    
    # 3) 키워드 분해 → FP/TU/SV 매핑
    def assign_aspect_keywords(row):
        if pd.isna(row["keyword"]): return "", "", ""
        kws = [k.strip() for k in row["keyword"].split(",") if k.strip()]
        if not kws: return "", "", ""
        emb = model.encode(kws)
        fp_sim = cosine_similarity(emb, [FP_mean]).flatten()
        tu_sim = cosine_similarity(emb, [TU_mean]).flatten()
        sv_sim = cosine_similarity(emb, [SV_mean]).flatten()
        fp_k, tu_k, sv_k = [], [], []
        for i, kw in enumerate(kws):
            sims = {"fp": fp_sim[i], "tu": tu_sim[i], "sv": sv_sim[i]}
            best = max(sims, key=sims.get)
            (fp_k if best == "fp" else tu_k if best == "tu" else sv_k).append(kw)
        return ", ".join(fp_k), ", ".join(tu_k), ", ".join(sv_k)

    df_join[["fp_keyword", "tu_keyword", "sv_keyword"]] = df_join.apply(assign_aspect_keywords, axis=1, result_type="expand")
    df_join.to_csv(os.path.join(JUSTIF_DIR, "base_finetuned_output_with_keywords.csv"), index=False)
    
    return df_join

def calculate_probability_delta(df_join, base_probs, finetuned_probs):
    """키워드별 확률 델타를 계산합니다."""
    # 4) 확률 델타 계산
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    records = []
    
    # 이전에 정의된 tokenize_keywords 함수를 사용
    def tok(kwstr):
        # 외부 함수인 tokenize_keywords를 호출
        return tokenize_keywords(tokenizer, kwstr)

    for _, row in df_join.iterrows():
        if pd.isna(row['keyword']):  # ★확률 델타는 오직 키워드가 있는 경우에만 계산
            continue

        i = row["Index"]
        if i not in base_probs or i not in finetuned_probs:
            continue
        
        # 아웃풋 토큰에서 각 키워드별 평균 확률을 계산
        base_map = build_token_prob_map_avg(base_probs[i], tokenizer)
        fine_map = build_token_prob_map_avg(finetuned_probs[i], tokenizer)
        
        # 키워드별로 델타 계산
        fp_delta = compute_delta(tok(row["fp_keyword"]), base_map, fine_map)
        tu_delta = compute_delta(tok(row["tu_keyword"]), base_map, fine_map)
        sv_delta = compute_delta(tok(row["sv_keyword"]), base_map, fine_map)
        
        records.append({"Index": i, "FP_delta": fp_delta, "TU_delta": tu_delta, "SV_delta": sv_delta})

    df_delta = pd.DataFrame(records).set_index("Index")
    print("Mean ΔP:", df_delta.mean().to_dict())
    
    return df_delta, tokenizer


def visualize_deltas(df_delta, save_figs):
    """확률 델타를 시각화합니다."""
    if save_figs:
        os.makedirs(FIG_DIR, exist_ok=True)
        
    # ΔP heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_delta, cmap="RdBu_r", center=0, linewidths=0.5)
    plt.title("Keyword Probability Delta (Fine-tuned – Base)")
    plt.xlabel("Aspect")
    plt.ylabel("Validation ID")
    plt.tight_layout()
    if save_figs:
        plt.savefig(os.path.join(FIG_DIR, "delta_heatmap.png"), dpi=180)
    plt.show()

    # ΔP KDE plot
    fp_deltas = df_delta['FP_delta'].dropna().values
    tu_deltas = df_delta['TU_delta'].dropna().values
    sv_deltas = df_delta['SV_delta'].dropna().values

    plt.figure(figsize=(10, 6))
    sns.kdeplot(fp_deltas, fill=True, label="FP", color="skyblue", linewidth=2)
    sns.kdeplot(tu_deltas, fill=True, label="TU", color="lightgreen", linewidth=2)
    sns.kdeplot(sv_deltas, fill=True, label="SV", color="salmon", linewidth=2)

    plt.axvline(0, linestyle='--', color='black', linewidth=1)
    plt.title("Probability Shift Distribution per Reasoning Dimension (ΔP)")
    plt.xlabel("Probability Change (Fine-tuned – Base)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    if save_figs:
        plt.savefig(os.path.join(FIG_DIR, "delta_kde.png"), dpi=180)
    plt.show()

def analyze_top_bottom_cases(df_join, df_delta):
    """가장 많이/적게 변화한 샘플을 분석하고 출력합니다."""
    # Δ 합산 기준 정렬
    df_delta["max_abs_delta"] = df_delta.abs().sum(axis=1)

    # 가장 많이 변한 Top 3
    top3_indices = df_delta["max_abs_delta"].nlargest(3).index

    # 가장 적게 변한 Bottom 3
    bottom3_indices = df_delta["max_abs_delta"].nsmallest(3).index

    # 🔷 Top 3 출력
    print("📈 🔷 [Top 3: 가장 많이 변화한 샘플]")
    for rank, idx in enumerate(top3_indices, 1):
        print(f"\n🟦 TOP {rank} — Index {idx} | Total Δ: {df_delta.loc[idx, 'max_abs_delta']:.4f}")
        print("🔹 [Base Reason]")
        print(df_join[df_join['Index'] == idx]['base_reason'].iloc[0])
        print("🔸 [Fine-tuned Reason]")
        print(df_join[df_join['Index'] == idx]['finetuned_reason'].iloc[0])

    # 🔹 Bottom 3 출력
    print("\n📉 🔹 [Bottom 3: 거의 변화 없는 샘플]")
    for rank, idx in enumerate(bottom3_indices, 1):
        print(f"\n⬜ BOTTOM {rank} — Index {idx} | Total Δ: {df_delta.loc[idx, 'max_abs_delta']:.4f}")
        print("🔹 [Base Reason]")
        print(df_join[df_join['Index'] == idx]['base_reason'].iloc[0])
        print("🔸 [Fine-tuned Reason]")
        print(df_join[df_join['Index'] == idx]['finetuned_reason'].iloc[0])

def visualize_delta_max(exam_idx, 
                         fp_keyword, tu_keyword, sv_keyword, 
                         tokenizer, base_probs, finetuned_probs,
                         save_figs):
    """확률 델타를 시각화합니다."""
    if save_figs:
        os.makedirs(FIG_DIR, exist_ok=True)

    fp_tokens = tokenize_keywords(tokenizer, fp_keyword)
    tu_tokens = tokenize_keywords(tokenizer, tu_keyword)
    sv_tokens = tokenize_keywords(tokenizer, sv_keyword)

    base_prob_map = build_token_prob_map_avg(base_probs[exam_idx], tokenizer)
    fine_prob_map = build_token_prob_map_avg(finetuned_probs[exam_idx], tokenizer)

    def extract_probs(token_list, prob_map):
        return [(token, prob_map.get(token, 0.0)) for token in token_list]

    fp_base = extract_probs(fp_tokens, base_prob_map)
    fp_fine = extract_probs(fp_tokens, fine_prob_map)
    tu_base = extract_probs(tu_tokens, base_prob_map)
    tu_fine = extract_probs(tu_tokens, fine_prob_map)
    sv_base = extract_probs(sv_tokens, base_prob_map)
    sv_fine = extract_probs(sv_tokens, fine_prob_map)

    df_tokens = pd.DataFrame({
        "Token": fp_tokens + tu_tokens + sv_tokens,
        "Group": ["FP"] * len(fp_tokens) + ["TU"] * len(tu_tokens) + ["SV"] * len(sv_tokens),
        "Base": [p[1] for p in fp_base] + [p[1] for p in tu_base] + [p[1] for p in sv_base],
        "Fine-tuned": [p[1] for p in fp_fine] + [p[1] for p in tu_fine] + [p[1] for p in sv_fine]
    })
    
    df_melted = df_tokens.melt(id_vars=["Token", "Group"], value_vars=["Base", "Fine-tuned"],
                               var_name="Model", value_name="Probability")
    
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df_melted, x="Token", y="Probability", hue="Model", palette="Set2")
    plt.title("Base vs Fine-tuned Token Probabilities for Keywords")
    plt.xlabel("Token")
    plt.ylabel("Avg Probability")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.show()

def main(save_figs=False):
    if save_figs:
        os.makedirs(FIG_DIR, exist_ok=True)

    # 1. 데이터 로드 및 전처리
    df_total, base_outputs, ft_outputs, df_keywords, base_probs, finetuned_probs = load_data_and_preprocess()

    # 2. 키워드 처리 및 데이터프레임 병합
    df_join = process_keywords_and_join(df_total, base_outputs, ft_outputs, df_keywords)

    # 3. 확률 델타 계산
    df_delta, tokenizer = calculate_probability_delta(df_join, base_probs, finetuned_probs)

    # 4. 시각화
    visualize_deltas(df_delta, save_figs)
    
    # 5. Top/Bottom 케이스 분석 및 출력
    analyze_top_bottom_cases(df_join, df_delta)

    # 6. ΔP most laregest case (별도로 시각화)
    exam_idx = 66
    fp_keyword = "cardiovascular disease monitoring"
    tu_keyword = "compact digital health device"
    sv_keyword = "improve treatment outcomes"

    visualize_delta_max(exam_idx, 
                         fp_keyword, tu_keyword, sv_keyword, 
                         tokenizer, base_probs, finetuned_probs,
                         save_figs)   
    

if __name__ == '__main__':
    main(save_figs=True)

