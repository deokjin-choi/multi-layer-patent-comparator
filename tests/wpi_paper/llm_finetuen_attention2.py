# %%
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch, torch.nn.functional as F
import pandas as pd, json, gc, os
import matplotlib.pyplot as plt
import numpy as np

base_model_path = "/mnt/d/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/7231864981174d9bee8c7687c24c8344414eae6b"
lora_path = "./lora_output/mistral_lora_no_quant"
offload_path = "./offload_dir"
os.makedirs(offload_path, exist_ok=True)

# ✅ tokenizer: 확률 코드와 동일
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # (확률 코드와 동일 세팅 유지)

# ✅ 4bit 양자화: 확률 코드와 동일
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

def load_base_model():
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
        local_files_only=True,
        attn_implementation="eager"  # ✅ eager 강제
    ).eval()
    model.config.output_attentions = True  # ✅ 어텐션 반환 활성화
    return model

def load_finetuned_model():
    base_for_lora = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
        local_files_only=True,
        attn_implementation="eager"  # ✅ eager 강제
    )
    base_for_lora.config.output_attentions = True  # ✅ 어텐션 반환 활성화

    model = PeftModel.from_pretrained(
        base_for_lora,
        lora_path,
        is_trainable=False,
        local_files_only=True
    ).eval()
    model.config.output_attentions = True  # ✅ 어텐션 반환 활성화
    return model

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

# ✅ attention 계산: 확률 코드와 동일한 generate 설정 + forward에 attention_mask 명시
def compute_attention_scores(model, prompt, target_words, target_token_ids_list):
    # 확률 코드와 동일: 마스크 없이 generate 가능 (deterministic)
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

    full_ids = gen.sequences  # [1, input+gen]
    gen_len = full_ids.shape[1] - input_len
    output_range = list(range(input_len, input_len + gen_len))

    # 결과 텍스트 (프롬프트 제외)
    generated_text = tokenizer.decode(full_ids[0][input_len:], skip_special_tokens=True).strip()
    print("Generated:", generated_text[:200])

    # ⚠️ 여기서부터가 'attention' 핵심: forward 재계산 + attention_mask 명시
    with torch.no_grad():
        full_attention_mask = torch.ones_like(full_ids).to(full_ids.device)  # 패딩 없음 → 전부 1
        outputs = model(
            input_ids=full_ids,
            attention_mask=full_attention_mask,
            output_attentions=True
        )
        # 레이어, 헤드 평균 = 확률 코드와 독립적 / 우리가 보고 싶은 글로벌 집중도
        attn_map = torch.stack(outputs.attentions).mean(dim=0).mean(dim=1)[0]  # (seq_len, seq_len)

    # 키워드별 attention 평균
    result = {}
    for word, token_ids in zip(target_words, target_token_ids_list):
        token_positions = []
        for tid in token_ids:
            pos = (full_ids[0] == tid).nonzero(as_tuple=True)[0]
            token_positions.extend(pos.tolist())
        if token_positions:
            score = attn_map[output_range][:, token_positions].mean().item() # 이게 평균이 되어야 함
            result[word] = round(score, 4)
        else:
            result[word] = 0.0
    return result

def process_model(df_keywords, valid_entries, use_lora=False, tag="base"):
    
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

        # keyword 전체가 NaN이면 skip
        if pd.isna(row.get("keyword")):
            continue

        fp_words = safe_split(row.get("fp_keyword"))
        tu_words = safe_split(row.get("tu_keyword"))
        sv_words = safe_split(row.get("sv_keyword"))
        all_words = fp_words + tu_words + sv_words
        token_ids_list = [tokenizer.encode(w, add_special_tokens=False) for w in all_words]

        # ✅ prompt 생성 (entry는 jsonl에서 가져온 것 그대로 사용)
        prompt = build_prompt_shared(entry)

        # ✅ 어텐션 점수 계산
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

# --- 새로운 함수들 추가 시작 ---
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

# --- 새로운 함수들 추가 끝 ---

if __name__ == "__main__":
    import seaborn as sns
    
    # ... (기존 실행 코드) ...
    # keyword 있는 것 가지고 오기
    df_keywords = pd.read_csv("valid_output_base_finetuend_with_keyword.csv")
    df_keywords.columns = df_keywords.columns.str.strip()

    # Load valid entries from JSONL file
    with open("fine_tuned_valid.jsonl", "r", encoding="utf-8") as f:
        valid_entries = [json.loads(line) for line in f]

    # Run
    df_base = process_model(df_keywords, valid_entries, use_lora=False, tag="base")
    #df_base.to_csv("attention_base.csv", index=False)
    
    # Step 2: Finetuned 모델 나중에 처리
    df_finetuned = process_model(df_keywords, valid_entries, use_lora=True, tag="finetuned")
    #df_finetuned.to_csv("attention_finetuned.csv", index=False)

    # Step 3: 두 결과 병합 및 저장
    df_merged = pd.merge(df_base, df_finetuned, on="id")
    #df_merged.to_csv("attention_comparison_final.csv", index=False)

    # 변화량 계산
    delta_df = df_merged.copy()
    delta_df["delta_fp"] = delta_df["finetuned_fp"] - delta_df["base_fp"]
    delta_df["delta_tu"] = delta_df["finetuned_tu"] - delta_df["base_tu"]
    delta_df["delta_sv"] = delta_df["finetuned_sv"] - delta_df["base_sv"]

    # 히트맵용 데이터
    heatmap_data = delta_df[["delta_fp", "delta_tu", "delta_sv"]]

    plt.figure(figsize=(6, 4))
    sns.heatmap(heatmap_data, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Attention Change (Fine-tuned - Base)")
    plt.show()

    print(heatmap_data.mean())

    # 🔹 FP, TU, SV 각각의 ΔAttention 분포 가져오기
    fp_deltas = delta_df['delta_fp'].dropna().values
    tu_deltas = delta_df['delta_tu'].dropna().values
    sv_deltas = delta_df['delta_sv'].dropna().values

    # 🔹 KDE Plot 시각화
    plt.figure(figsize=(10, 6))
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
    plt.show()
    
# ----------------------------------------------------
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

    # Base 모델을 다시 로드하여 심층 분석
    base_model_for_analysis = load_base_model()
    base_outputs, base_full_ids, base_output_range = get_attention_outputs(base_model_for_analysis, prompt)
    base_layer_scores = get_layer_attention_scores(base_outputs, base_full_ids, base_output_range, all_words, token_ids_list)
    
    # Base 모델 삭제 후 메모리 확보
    del base_model_for_analysis
    gc.collect()
    torch.cuda.empty_cache()

    # Finetuned 모델을 다시 로드하여 심층 분석
    finetuned_model_for_analysis = load_finetuned_model()
    finetuned_outputs, finetuned_full_ids, finetuned_output_range = get_attention_outputs(finetuned_model_for_analysis, prompt)
    finetuned_layer_scores = get_layer_attention_scores(finetuned_outputs, finetuned_full_ids, finetuned_output_range, all_words, token_ids_list)
    
    # Finetuned 모델 삭제 후 메모리 확보
    del finetuned_model_for_analysis
    gc.collect()
    torch.cuda.empty_cache()

    # 레이어별 그래프 시각화
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

    # 개별 키워드 라벨에 고유한 색상 맵핑
    color_map = {}
    colors = plt.get_cmap('tab10').colors 
    for i, label in enumerate(simplified_labels):
        color_map[label] = colors[i % len(colors)]
        
    # 레이어별 그래프 시각화
    num_layers = len(finetuned_outputs.attentions)
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
    plt.show()
        
    # ----------------------------------------------------
    print("-" * 50)
    print("--- 2. 특정 레이어의 헤드별 어텐션 변화 시각화 시작 ---")

    # ⭐ 레이어별 그래프를 보고 가장 변화가 큰 레이어를 동적으로 선택
    # base와 finetuned 모델 간의 어텐션 점수 차이(절대값) 합이 가장 큰 레이어를 찾음
    layer_diffs = []
    for i in range(num_layers):
        layer_diff = 0
        for word in all_words:
            diff = abs(base_layer_scores[word][i] - finetuned_layer_scores[word][i])
            layer_diff += diff
        layer_diffs.append(layer_diff)
    
    target_layer_idx = np.argmax(layer_diffs)

    # Base 모델 재로드
    base_model_for_analysis = load_base_model()
    base_outputs, base_full_ids, base_output_range = get_attention_outputs(base_model_for_analysis, prompt)
    base_head_scores = get_head_attention_scores(base_outputs, base_full_ids, base_output_range, target_layer_idx, all_words, token_ids_list)
    del base_model_for_analysis
    gc.collect()
    torch.cuda.empty_cache()

    # Finetuned 모델 재로드
    finetuned_model_for_analysis = load_finetuned_model()
    finetuned_outputs, finetuned_full_ids, finetuned_output_range = get_attention_outputs(finetuned_model_for_analysis, prompt)
    finetuned_head_scores = get_head_attention_scores(finetuned_outputs, finetuned_full_ids, finetuned_output_range, target_layer_idx, all_words, token_ids_list)
    del finetuned_model_for_analysis
    gc.collect()
    torch.cuda.empty_cache()

    # 헤드별 바 차트 시각화
    num_heads = len(finetuned_outputs.attentions[target_layer_idx][0])
    x = np.arange(num_heads)
    width = 0.25

    fig, ax = plt.subplots(figsize=(18, 8))

    ax.bar(x - width, base_head_scores['monitoring cardiovascular diseases in lay user environments'], width, label='Base FP', color='skyblue')
    ax.bar(x, finetuned_head_scores['monitoring cardiovascular diseases in lay user environments'], width, label='Finetuned FP', color='darkblue')

    ax.bar(x + width, base_head_scores['sensors'], width, label='Base TU', color='lightgreen')
    ax.bar(x + 2*width, finetuned_head_scores['sensors'], width, label='Finetuned TU', color='darkgreen')

    ax.bar(x + 3*width, base_head_scores['direct health benefits'], width, label='Base SV', color='lightcoral')
    ax.bar(x + 4*width, finetuned_head_scores['direct health benefits'], width, label='Finetuned SV', color='darkred')

    ax.set_ylabel('Average Attention Score')
    ax.set_xlabel('Head Index')
    ax.set_title(f'Attention Score by Head for Layer {target_layer_idx}')
    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels(x)
    ax.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    print("-" * 50)