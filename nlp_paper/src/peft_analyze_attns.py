#!/usr/bin/env python
import os, json, pickle, gc
import numpy as np
import pandas as pd
from collections import defaultdict
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import matplotlib.pyplot as plt
import seaborn as sns

# 프로젝트 디렉토리 설정
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
NLP_PAPER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 경로
ORIGIN_CSV = os.path.join(NLP_PAPER_DIR, "data/prompts/origin/trial_full.csv")
REVISED_CSV = os.path.join(NLP_PAPER_DIR, "data/prompts/revised/trial_full.csv")
JUSTIF_DIR = os.path.join(NLP_PAPER_DIR, "data/peft/justifications")
ATTN_DIR = os.path.join(NLP_PAPER_DIR, "data/peft/attention")
FIG_DIR = os.path.join(NLP_PAPER_DIR, "data/peft/figs")
BASE_MODEL_PATH = "/mnt/d/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/7231864981174d9bee8c7687c24c8344414eae6b"
LORA_PATH = os.path.join(NLP_PAPER_DIR, "data/peft/lora_output/mistral_lora_quant")

# 전역 변수로 모델 및 토크나이저 설정
# 4bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 토크나이저: 확률 코드와 동일
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def load_base_model():
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
        local_files_only=True,
        attn_implementation="eager"
    ).eval()
    model.config.output_attentions = True
    return model

def load_finetuned_model():
    base_for_lora = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
        local_files_only=True,
        attn_implementation="eager"
    )
    base_for_lora.config.output_attentions = True

    model = PeftModel.from_pretrained(
        base_for_lora,
        LORA_PATH,
        is_trainable=False,
        local_files_only=True
    ).eval()
    model.config.output_attentions = True
    return model

def load_and_merge_data(origin_csv, revised_csv):
    d1 = pd.read_csv(origin_csv)
    d2 = pd.read_csv(revised_csv)
    return pd.concat([d1, d2], ignore_index=True)

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

def compute_attention_scores(model, prompt, target_words, target_token_ids_list):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    gen = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True
    )

    full_ids = gen.sequences
    gen_len = full_ids.shape[1] - input_len
    output_range = list(range(input_len, input_len + gen_len))

    generated_text = tokenizer.decode(full_ids[0][input_len:], skip_special_tokens=True).strip()
    print("Generated:", generated_text[:200])

    with torch.no_grad():
        full_attention_mask = torch.ones_like(full_ids).to(full_ids.device)
        outputs = model(
            input_ids=full_ids,
            attention_mask=full_attention_mask,
            output_attentions=True
        )
        attn_map = torch.stack(outputs.attentions).mean(dim=0).mean(dim=1)[0]

    result = {}
    for word, token_ids in zip(target_words, target_token_ids_list):
        token_positions = []
        for tid in token_ids:
            pos = (full_ids[0] == tid).nonzero(as_tuple=True)[0]
            token_positions.extend(pos.tolist())
        if token_positions:
            score = attn_map[output_range][:, token_positions].mean().item()
            result[word] = round(score, 4)
        else:
            result[word] = 0.0
    return result

def process_model_attentions(df_keywords, valid_entries, use_lora=False, tag="base"):
    if use_lora:
        model = load_finetuned_model()
    else:
        model = load_base_model()

    all_results = []
    def safe_split(val):
        return [w.strip() for w in str(val).split(",") if w.strip()] if pd.notna(val) else []

    for idx, entry in enumerate(valid_entries):
        print(f"Processing row {idx + 1}/{len(valid_entries)}")
        row = df_keywords.iloc[idx]
        if pd.isna(row.get("keyword")):
            continue

        fp_words = safe_split(row.get("fp_keyword"))
        tu_words = safe_split(row.get("tu_keyword"))
        sv_words = safe_split(row.get("sv_keyword"))
        all_words = fp_words + tu_words + sv_words
        token_ids_list = [tokenizer.encode(w, add_special_tokens=False) for w in all_words]

        prompt = build_prompt_shared(entry)
        scores = compute_attention_scores(model, prompt, all_words, token_ids_list)

        def avg_score(keywords):
            vals = [scores[w] for w in keywords if w in scores]
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        result = {
            "id": idx,
            f"{tag}_fp": avg_score(fp_words),
            f"{tag}_tu": avg_score(tu_words),
            f"{tag}_sv": avg_score(sv_words),
        }
        print(f"Results for row {idx + 1}: {result}")
        all_results.append(result)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return pd.DataFrame(all_results)

def get_attention_outputs(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    gen = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True
    )
    full_ids = gen.sequences
    gen_len = full_ids.shape[1] - input_len
    output_range = list(range(input_len, input_len + gen_len))
    
    with torch.no_grad():
        outputs = model(
            input_ids=full_ids,
            attention_mask=torch.ones_like(full_ids).to(full_ids.device),
            output_attentions=True
        )
    return outputs, full_ids, output_range

def get_layer_attention_scores(outputs, full_ids, output_range, target_words, target_token_ids_list):
    attn_by_layer = outputs.attentions
    num_layers = len(attn_by_layer)
    layer_scores = {word: [] for word in target_words}

    for i in range(num_layers):
        attn_map_layer = attn_by_layer[i][0].mean(dim=0)
        for word, token_ids in zip(target_words, target_token_ids_list):
            token_positions = []
            for tid in token_ids:
                pos = (full_ids[0] == tid).nonzero(as_tuple=True)[0]
                token_positions.extend(pos.tolist())
            
            if token_positions:
                attention_score = attn_map_layer[output_range][:, token_positions].mean().item()
                layer_scores[word].append(attention_score)
            else:
                layer_scores[word].append(0.0)
    return layer_scores

def get_head_attention_scores(outputs, full_ids, output_range, target_layer_idx, target_words, target_token_ids_list):
    attn_by_layer = outputs.attentions
    head_scores = {word: [] for word in target_words}
    num_heads = attn_by_layer[target_layer_idx].shape[1]

    for head_idx in range(num_heads):
        attn_map_head = attn_by_layer[target_layer_idx][0][head_idx]
        for word, token_ids in zip(target_words, target_token_ids_list):
            token_positions = []
            for tid in token_ids:
                pos = (full_ids[0] == tid).nonzero(as_tuple=True)[0]
                token_positions.extend(pos.tolist())
            
            if token_positions:
                attention_score = attn_map_head[output_range][:, token_positions].mean().item()
                head_scores[word].append(attention_score)
            else:
                head_scores[word].append(0.0)
    return head_scores

def main():
    os.makedirs(ATTN_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # 데이터 로드
    valid_path = os.path.join(NLP_PAPER_DIR, "data/peft/fine_tuned_valid.jsonl")
    keywords_path = os.path.join(JUSTIF_DIR, "base_finetuned_output_with_keywords.csv")
    
    with open(valid_path, "r", encoding="utf-8") as f:
        valid_entries = [json.loads(line) for line in f]
    df_keywords = pd.read_csv(keywords_path)

    # Step 1: Base 모델 어텐션 점수 계산 및 저장
    print("✅ Base Model Attention Scores Calculation Started.")
    df_base = process_model_attentions(df_keywords, valid_entries, use_lora=False, tag="base_attn")
    df_base.to_csv(os.path.join(ATTN_DIR, "attention_base.csv"), index=False)
    
    # Step 2: Finetuned 모델 어텐션 점수 계산 및 저장
    print("✅ Fine-tuned Model Attention Scores Calculation Started.")
    df_finetuned = process_model_attentions(df_keywords, valid_entries, use_lora=True, tag="ft_attn")
    df_finetuned.to_csv(os.path.join(ATTN_DIR, "attention_finetuned.csv"), index=False)

    # Step 3: 두 결과 병합 및 저장
    df_merged = pd.merge(df_base, df_finetuned, on="id")
    df_merged.to_csv(os.path.join(ATTN_DIR, "attention_comparison_final.csv"), index=False)

    # 변화량 계산
    delta_df = df_merged.copy()
    delta_df["delta_fp"] = delta_df["ft_attn_fp"] - delta_df["base_attn_fp"]
    delta_df["delta_tu"] = delta_df["ft_attn_tu"] - delta_df["base_attn_tu"]
    delta_df["delta_sv"] = delta_df["ft_attn_sv"] - delta_df["base_attn_sv"]

    print("\n--- Attention Analysis Complete ---")
    print("Mean ΔAttention:", delta_df[["delta_fp", "delta_tu", "delta_sv"]].mean().to_dict())

    # 시각화 1: 히트맵
    plt.figure(figsize=(10, 8))
    sns.heatmap(delta_df[["delta_fp", "delta_tu", "delta_sv"]], cmap="coolwarm", center=0, cbar=True)
    plt.title("Attention Change (Fine-tuned - Base)")
    plt.xlabel("Reasoning Dimension")
    plt.ylabel("Data Index")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "delta_heatmap.png"))
    plt.close()
    
    # 시각화 2: KDE Plot
    plt.figure(figsize=(10, 6))
    fp_deltas = delta_df['delta_fp'].dropna().values
    tu_deltas = delta_df['delta_tu'].dropna().values
    sv_deltas = delta_df['delta_sv'].dropna().values
    sns.kdeplot(fp_deltas, fill=True, label="FP", color="skyblue", linewidth=2)
    sns.kdeplot(tu_deltas, fill=True, label="TU", color="lightgreen", linewidth=2)
    sns.kdeplot(sv_deltas, fill=True, label="SV", color="salmon", linewidth=2)
    plt.axvline(0, linestyle='--', color='black', linewidth=1)
    plt.title("Attention Score Change Distribution per Reasoning Dimension (ΔAttention)")
    plt.xlabel("Attention Change (Fine-tuned – Base)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "delta_kde.png"))
    plt.close()

    # Step 5: 심층 분석 (가장 변화가 큰 케이스 1개)
    print("-" * 50)
    print("--- 1. 레이어별 어텐션 변화 시각화 시작 ---")
    delta_df['delta_sum_abs'] = delta_df['delta_fp'].abs() + delta_df['delta_tu'].abs() + delta_df['delta_sv'].abs()
    analysis_idx = delta_df['delta_sum_abs'].idxmax()
    selected_entry = valid_entries[analysis_idx]
    selected_row = df_keywords.iloc[analysis_idx]
    print(f"가장 큰 변화를 보인 인덱스(ID): {analysis_idx}를 선택하여 분석을 시작합니다.")

    def safe_split(val):
        return [w.strip() for w in str(val).split(",") if w.strip()] if pd.notna(val) else []
    
    fp_words = safe_split(selected_row.get("fp_keyword"))
    tu_words = safe_split(selected_row.get("tu_keyword"))
    sv_words = safe_split(selected_row.get("sv_keyword"))
    all_words = fp_words + tu_words + sv_words
    token_ids_list = [tokenizer.encode(w, add_special_tokens=False) for w in all_words]
    prompt = build_prompt_shared(selected_entry)

    base_model_for_analysis = load_base_model()
    base_outputs, base_full_ids, base_output_range = get_attention_outputs(base_model_for_analysis, prompt)
    base_layer_scores = get_layer_attention_scores(base_outputs, base_full_ids, base_output_range, all_words, token_ids_list)
    del base_model_for_analysis; gc.collect(); torch.cuda.empty_cache()

    finetuned_model_for_analysis = load_finetuned_model()
    finetuned_outputs, finetuned_full_ids, finetuned_output_range = get_attention_outputs(finetuned_model_for_analysis, prompt)
    finetuned_layer_scores = get_layer_attention_scores(finetuned_outputs, finetuned_full_ids, finetuned_output_range, all_words, token_ids_list)
    del finetuned_model_for_analysis; gc.collect(); torch.cuda.empty_cache()

    num_layers = len(finetuned_outputs.attentions)
    simplified_labels = []
    fp_counter, tu_counter, sv_counter = 1, 1, 1
    for word in all_words:
        if word in fp_words:
            simplified_labels.append(f'FP{fp_counter}')
            fp_counter += 1
        elif word in tu_words:
            simplified_labels.append(f'TU{tu_counter}')
            tu_counter += 1
        elif word in sv_words:
            simplified_labels.append(f'SV{sv_counter}')
            sv_counter += 1
        else:
            simplified_labels.append(word)
    color_map = {}
    colors = plt.get_cmap('tab10').colors
    for i, label in enumerate(simplified_labels):
        color_map[label] = colors[i % len(colors)]
        
    plt.figure(figsize=(15, 6))
    for i, word in enumerate(all_words):
        label = simplified_labels[i]
        color = color_map.get(label, 'gray')
        plt.plot(range(num_layers), base_layer_scores[word], linestyle='--', label=f'Base {label}', color=color)
        plt.plot(range(num_layers), finetuned_layer_scores[word], label=f'Finetuned {label}', marker='o', markersize=4, color=color)
    plt.title(f'Attention Score by Layer in a healthcare example')
    plt.xlabel('Layer Index')
    plt.ylabel('Average Attention Score')
    plt.xticks(range(0, num_layers, 2))
    plt.legend(loc='upper right', ncol=3)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print("-" * 50)
    print("--- 2. 특정 레이어의 헤드별 어텐션 변화 시각화 시작 ---")
    layer_diffs = []
    for i in range(num_layers):
        layer_diff = 0
        for word in all_words:
            diff = abs(base_layer_scores[word][i] - finetuned_layer_scores[word][i])
            layer_diff += diff
        layer_diffs.append(layer_diff)
    target_layer_idx = np.argmax(layer_diffs)
    print(f"가장 큰 변화를 보인 레이어: {target_layer_idx}")
    
    base_model_for_analysis = load_base_model()
    base_outputs, base_full_ids, base_output_range = get_attention_outputs(base_model_for_analysis, prompt)
    base_head_scores = get_head_attention_scores(base_outputs, base_full_ids, base_output_range, target_layer_idx, all_words, token_ids_list)
    del base_model_for_analysis; gc.collect(); torch.cuda.empty_cache()

    finetuned_model_for_analysis = load_finetuned_model()
    finetuned_outputs, finetuned_full_ids, finetuned_output_range = get_attention_outputs(finetuned_model_for_analysis, prompt)
    finetuned_head_scores = get_head_attention_scores(finetuned_outputs, finetuned_full_ids, finetuned_output_range, target_layer_idx, all_words, token_ids_list)
    del finetuned_model_for_analysis; gc.collect(); torch.cuda.empty_cache()

    num_heads = len(finetuned_outputs.attentions[target_layer_idx][0])
    x = np.arange(num_heads)
    width = 0.25

    fig, ax = plt.subplots(figsize=(18, 8))
    fp_keywords_plot = [kw for kw in all_words if kw in fp_words]
    tu_keywords_plot = [kw for kw in all_words if kw in tu_words]
    sv_keywords_plot = [kw for kw in all_words if kw in sv_words]

    for i, word in enumerate(fp_keywords_plot):
        ax.bar(x - (len(fp_keywords_plot)-1)*width/2 + i*width, base_head_scores[word], width, label=f'Base FP{i+1}')
        ax.bar(x - (len(fp_keywords_plot)-1)*width/2 + i*width, finetuned_head_scores[word], width, label=f'Finetuned FP{i+1}')

    ax.set_ylabel('Average Attention Score')
    ax.set_xlabel('Head Index')
    ax.set_title(f'Attention Score by Head for Layer {target_layer_idx}')
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    print("-" * 50)

if __name__ == "__main__":
    main()