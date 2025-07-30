# %%
import json
import torch
import csv
import pickle
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from scipy.special import rel_entr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# 🔧 경로 설정
base_model_path = "/mnt/d/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/7231864981174d9bee8c7687c24c8344414eae6b"
fine_tuned_model_path = "./lora_output/mistral_lora_no_quant"
valid_file_path = "fine_tuned_valid.jsonl"
output_file_path = "kl_results_with_generations.jsonl"

# 🔧 tokenizer 및 quantization config
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 🔧 프롬프트 생성 함수
def build_prompt(entry):
    input_ = entry["input"]
    prompt = (
        f"### Input:\n"
        f"fp_winner: {input_['fp_winner']}\n"
        f"fp_reason: {input_['fp_reason']}\n"
        f"tu_winner: {input_['tu_winner']}\n"
        f"tu_reason: {input_['tu_reason']}\n"
        f"sv_winner: {input_['sv_winner']}\n"
        f"sv_reason: {input_['sv_reason']}\n"
        f"overall_winner: {input_['overall_winner']}\n\n"
        f"### Output:\n"
        f"Write an overall judgement (1–2 sentences) justifying the overall winner. \n"
        f"Do not start with sentences like 'The overall winner is...'\n"
        f"This explanation must reflect the actual aspect(s) that contributed to the win (e.g., functional purpose, technical uniqueness, or strategic value).\n"
    )
    return prompt

def build_prompt_for_base_model(entry):
    input_ = entry["input"]
    return (
        f"Below is a comparison between two patents: Our patent and the Competitor's patent.\n\n"
        f"Functional Purpose:\n"
        f"- Winner: {input_['fp_winner']}\n"
        f"- Reason: {input_['fp_reason']}\n\n"
        f"Technical Uniqueness:\n"
        f"- Winner: {input_['tu_winner']}\n"
        f"- Reason: {input_['tu_reason']}\n\n"
        f"Strategic Value:\n"
        f"- Winner: {input_['sv_winner']}\n"
        f"- Reason: {input_['sv_reason']}\n\n"
        f"overall_winner: {input_['overall_winner']}\n\n"
        f"Write an overall judgement (1–2 sentences) justifying the overall winner. \n"

    )

def build_prompt_shared(entry):
    input_ = entry["input"]
    return (
        f"Below is a comparison between two patents: Our patent and the Competitor's patent.\n\n"
        f"Functional Purpose:\n"
        f"- Winner: {input_['fp_winner']}\n"
        f"- Reason: {input_['fp_reason']}\n\n"
        f"Technical Uniqueness:\n"
        f"- Winner: {input_['tu_winner']}\n"
        f"- Reason: {input_['tu_reason']}\n\n"
        f"Strategic Value:\n"
        f"- Winner: {input_['sv_winner']}\n"
        f"- Reason: {input_['sv_reason']}\n\n"
        f"overall_winner: {input_['overall_winner']}\n\n"
        f"Write an overall judgement (1–2 sentences) justifying the overall winner.\n"
        f"Do not start with sentences like 'The overall winner is...'.\n"
        f"This explanation must reflect the actual aspect(s) that contributed to the win "
        f"(e.g., functional purpose, technical uniqueness, or strategic value).\n"
    )

def load_and_merge_data(old_path, new_path):
    df_old = pd.read_csv(old_path)
    df_new = pd.read_csv(new_path)
    df_all = pd.concat([df_old, df_new], ignore_index=True)
    return df_all

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


# 🔧 KL divergence 계산 함수
def compute_kl(p_list, q_list):
    kls = []
    for p, q in zip(p_list, q_list):
        p = np.array(p)
        q = np.array(q)
        kls.append(np.sum(rel_entr(p, q)))
    return np.mean(kls)

# 🔧 검증 데이터 로드
with open(valid_file_path, "r", encoding="utf-8") as f:
    valid_entries = [json.loads(line) for line in f]

TOP_K = 100

# %%
# 🔹 Step 1: Base Model 추론 및 확률 저장
print("🚀 Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
    local_files_only=True
)
base_model.eval()

# %%
print("📌 Generating base model outputs and probabilities...")
base_probs = {}
base_outputs = {}

for i, entry in enumerate(tqdm(valid_entries, desc="Base Model")):
    prompt = build_prompt_shared(entry)
    inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = base_model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            #temperature=0.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )

    token_probs = []

    # outputs.scores: list of tensors [batch_size=1, vocab_size]
    for score in outputs.scores:
        probs = F.softmax(score, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, k=TOP_K, dim=-1)

        # topk_indices[0]과 topk_probs[0]은 각각 [TOP_K] 형태
        token_dict = {
            int(idx): float(prob)
            for idx, prob in zip(topk_indices[0], topk_probs[0])
        }

        token_probs.append(token_dict)
    
    base_probs[i] = token_probs

    gen_tokens = outputs.sequences[0][input_length:]
    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    print(f"Generated text for entry {i}: {gen_text}")
    base_outputs[i] = gen_text

# base_outputs 저장
'''
with open("base_outputs2.csv", "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Index", "overall_reason"])
    for i, text in base_outputs.items():
        writer.writerow([i, text])

# ✅ base_probs를 Pickle로 저장
with open("base_probs2.pkl", "wb") as f:
    pickle.dump(base_probs, f)

'''

del base_model
torch.cuda.empty_cache()

# %%
# 🔹 Step 2: Fine-tuned Model 추론 및 확률 저장
print("🚀 Loading fine-tuned model...")
base_for_lora = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
    local_files_only=True
)

fine_tuned_model = PeftModel.from_pretrained(
    base_for_lora,
    fine_tuned_model_path,
    is_trainable=False,
    local_files_only=True
)
fine_tuned_model.eval()

print("📌 Generating fine-tuned model outputs and probabilities...")
finetuned_probs = {}
finetuned_outputs = {}

for i, entry in enumerate(tqdm(valid_entries, desc="Fine-tuned Model")):
    prompt = build_prompt_shared(entry)
    inputs = tokenizer(prompt, return_tensors="pt").to(fine_tuned_model.device)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = fine_tuned_model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            #temperature=0.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )

    token_probs = []
    for score in outputs.scores:
        probs = F.softmax(score, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, k=TOP_K, dim=-1)

        token_dict = {
            int(idx): float(prob)
            for idx, prob in zip(topk_indices[0], topk_probs[0])
        }
        token_probs.append(token_dict)

    finetuned_probs[i] = token_probs

    gen_tokens = outputs.sequences[0][input_length:]
    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    print(f"Generated text for entry {i}: {gen_text}")
    finetuned_outputs[i] = gen_text

with open("finetuned_outputs2.csv", "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Index", "overall_reason"])
    for i, text in finetuned_outputs.items():
        writer.writerow([i, text])

# finetuned_probs 저장 (확률 리스트를 문자열로 저장)
with open("finetuned_probs2.pkl", "wb") as f:
    pickle.dump(finetuned_probs, f)

del fine_tuned_model
torch.cuda.empty_cache()
# %%
df_total = load_and_merge_data("old/trial_full.csv", "new/trial_full.csv")
df_base = pd.read_csv("base_outputs2.csv")
df_finetuned = pd.read_csv("finetuned_outputs2.csv")

# 줄 단위로 읽되, 비어있는 줄은 np.nan으로
with open("base_finetuned_keyword.txt", "r", encoding="utf-8") as f:
    keyword_lines = [line.strip() if line.strip() else np.nan for line in f]

# DataFrame으로 변환
df_keywords = pd.DataFrame({'keyword': keyword_lines})

# base 모델에 출력 에러가 있는 것 제외
df_base = df_base.join(df_keywords)
df_base_notna = df_base[df_base['keyword'].notna()]
df_finetuned = df_finetuned.join(df_keywords)

df_base_notna = add_dominant_reason(df_total, df_base_notna, model_name="AI-Growth-Lab/PatentSBerta")
df_finetuned = add_dominant_reason(df_total, df_finetuned, model_name="AI-Growth-Lab/PatentSBerta")

# Reason Dominant Aspect 분포 출력
print("Base Model Dominant Aspect Distribution:")
print(df_base_notna['reason_dominant_aspect'].value_counts(normalize=True))
print("\nFine-tuned Model Dominant Aspect Distribution:")
print(df_finetuned['reason_dominant_aspect'].value_counts(normalize=True))


# 이후에 비교할 df_base와 df_finetuned를 통합 -> 각 행별 keyword 확률분포 파악
df_base.rename(columns={'overall_reason': 'base_reason'}, inplace=True)
df_finetuned.rename(columns={'overall_reason': 'finetuned_reason'}, inplace=True)

df_base_finetuned = df_base[['base_reason', 'keyword','Index']].merge(df_finetuned[['finetuned_reason','Index']], how='left', on='Index').drop('Index', axis=1)
df_base_finetuned = df_base_finetuned[['base_reason', 'finetuned_reason','keyword']]

df_base_finetuned
# %%

# 각 이유별 중심점 게산
def compute_mean_embeddings(df_total, model_name="AI-Growth-Lab/PatentSBerta"):
    model = SentenceTransformer(model_name)

    fp_vecs = model.encode(df_total["fp_reason"].fillna("").tolist(), show_progress_bar=True)
    tu_vecs = model.encode(df_total["tu_reason"].fillna("").tolist(), show_progress_bar=True)
    sv_vecs = model.encode(df_total["sv_reason"].fillna("").tolist(), show_progress_bar=True)

    FP_mean = np.mean(fp_vecs, axis=0)
    TU_mean = np.mean(tu_vecs, axis=0)
    SV_mean = np.mean(sv_vecs, axis=0)

    return model, FP_mean, TU_mean, SV_mean


# 각 행별 키워드 분할해서 fp,tu, sv로 분류
def assign_aspect_to_keywords(df_base_finetuned, model, FP_mean, TU_mean, SV_mean):
    fp_keywords, tu_keywords, sv_keywords = [], [], []

    for row in df_base_finetuned['keyword']:
        if pd.isna(row):
            # NaN일 경우 빈 값 추가 후 다음으로 -> 이후 skip처리
            fp_keywords.append("")
            tu_keywords.append("")
            sv_keywords.append("")
            continue

        keywords = [k.strip() for k in row.split(',') if k.strip()]
        if not keywords:
            fp_keywords.append("")
            tu_keywords.append("")
            sv_keywords.append("")
            continue

        embeddings = model.encode(keywords)

        # 유사도 계산
        fp_sim = cosine_similarity(embeddings, [FP_mean]).flatten()
        tu_sim = cosine_similarity(embeddings, [TU_mean]).flatten()
        sv_sim = cosine_similarity(embeddings, [SV_mean]).flatten()

        fp_k, tu_k, sv_k = [], [], []
        for i, keyword in enumerate(keywords):
            sims = {'fp': fp_sim[i], 'tu': tu_sim[i], 'sv': sv_sim[i]}
            best_match = max(sims, key=sims.get)
            if best_match == 'fp':
                fp_k.append(keyword)
            elif best_match == 'tu':
                tu_k.append(keyword)
            elif best_match == 'sv':
                sv_k.append(keyword)

        fp_keywords.append(", ".join(fp_k))
        tu_keywords.append(", ".join(tu_k))
        sv_keywords.append(", ".join(sv_k))

    df_base_finetuned['fp_keyword'] = fp_keywords
    df_base_finetuned['tu_keyword'] = tu_keywords
    df_base_finetuned['sv_keyword'] = sv_keywords

    return df_base_finetuned


model, FP_mean, TU_mean, SV_mean = compute_mean_embeddings(df_total)
df_base_finetuned_keyword_split = assign_aspect_to_keywords(df_base_finetuned.copy(), model, FP_mean, TU_mean, SV_mean)

# %%
import pickle
import torch
from transformers import AutoTokenizer
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 🔹 확률 정보 로드
with open("base_probs2.pkl", "rb") as f:
    base_probs = pickle.load(f)
with open("finetuned_probs2.pkl", "rb") as f:
    finetuned_probs = pickle.load(f)

# 🔹 데이터프레임 로드

# ✅ 확률 맵 생성 함수
def build_token_prob_map_avg(prob_dicts):
    token_sums = defaultdict(float)
    token_counts = defaultdict(int)

    for prob_dict in prob_dicts:
        for token_id, prob in prob_dict.items():
            token_str = tokenizer.convert_ids_to_tokens(token_id)
            token_sums[token_str] += prob
            token_counts[token_str] += 1

    return {token: token_sums[token] / token_counts[token] for token in token_sums}

# ✅ Δ 계산 함수
def compute_delta(tokens, base_map, fine_map):
    deltas = []
    for token in tokens:
        base_p = base_map.get(token, 0.0)
        fine_p = fine_map.get(token, 0.0)
        deltas.append(fine_p - base_p)
    return np.mean(deltas) if deltas else 0.0

# ✅ 결과 저장용
records = []

# ✅ 각 샘플 처리
for i, row in df_base_finetuned_keyword_split.iterrows():
    if pd.isna(row['keyword']):
        continue

    # 🔹 키워드 토큰화
    def tokenize(phrase_str):
        phrases = [p.strip() for p in str(phrase_str).split(',') if p.strip()]
        tokens = []
        for phrase in phrases:
            tokens.extend(tokenizer.tokenize(phrase))
        return tokens

    fp_tokens = tokenize(row['fp_keyword'])
    tu_tokens = tokenize(row['tu_keyword'])
    sv_tokens = tokenize(row['sv_keyword'])

    # 🔹 확률 맵 생성
    base_prob_map = build_token_prob_map_avg(base_probs[i])
    fine_prob_map = build_token_prob_map_avg(finetuned_probs[i])

    # 🔹 Δ 계산
    fp_delta = compute_delta(fp_tokens, base_prob_map, fine_prob_map)
    tu_delta = compute_delta(tu_tokens, base_prob_map, fine_prob_map)
    sv_delta = compute_delta(sv_tokens, base_prob_map, fine_prob_map)

    records.append({
        'id': i,
        'FP_delta': fp_delta,
        'TU_delta': tu_delta,
        'SV_delta': sv_delta
    })

# ✅ DataFrame 생성
df_delta = pd.DataFrame(records).set_index('id')

# ✅ 히트맵 출력
plt.figure(figsize=(10, 6))
sns.heatmap(df_delta, cmap="RdBu_r", center=0, linewidths=0.5)
plt.title("FP / TU / SV Keyword Probability Δ per Sample")
plt.xlabel("Aspect")
plt.ylabel("Validation ID")
plt.tight_layout()
plt.show()

# %%
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
    print(df_base_finetuned_keyword_split.loc[idx, 'base_reason'])
    print("🔸 [Fine-tuned Reason]")
    print(df_base_finetuned_keyword_split.loc[idx, 'finetuned_reason'])

# 🔹 Bottom 3 출력
print("\n📉 🔹 [Bottom 3: 거의 변화 없는 샘플]")
for rank, idx in enumerate(bottom3_indices, 1):
    print(f"\n⬜ BOTTOM {rank} — Index {idx} | Total Δ: {df_delta.loc[idx, 'max_abs_delta']:.4f}")
    print("🔹 [Base Reason]")
    print(df_base_finetuned_keyword_split.loc[idx, 'base_reason'])
    print("🔸 [Fine-tuned Reason]")
    print(df_base_finetuned_keyword_split.loc[idx, 'finetuned_reason'])
# %%

